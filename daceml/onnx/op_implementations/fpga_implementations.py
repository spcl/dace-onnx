import copy
import inspect
import typing

import dace
from dace import SDFGState, SDFG, dtypes
from dace.frontend.python.parser import DaceProgram
from dace.registry import autoregister_params
from dace.sdfg import nodes, propagation
from dace.sdfg.nodes import Node
from dace.symbolic import symstr

from daceml.onnx.nodes.onnx_op import ONNXOp
from daceml.onnx import converters
from daceml.onnx.implementation_abc import ONNXForward
import numpy as np
import math

from daceml.util.utils import in_desc_with_name, out_desc_with_name


def _2d_sliding_window_index_expr(x_or_y, stride, kernel_size):
    index_expression = "out_{x_or_y} * {stride} + h{x_or_y}"
    return index_expression.format(x_or_y=x_or_y, stride=stride)


def program_for_node(program, sdfg: SDFG, state: SDFGState,
                     node: ONNXOp) -> DaceProgram:
    """ Expand a function to a dace program.

        The dtypes for the arguments will be extracted by matching the parameter names to edges.
    """
    input_names = set(inp.name for inp in node.schema.inputs)
    output_names = set(outp.name for outp in node.schema.outputs)

    if input_names.intersection(output_names):
        # this is currently the case for only one onnx op
        raise ValueError(
            "program_for_node cannot be applied on nodes of this type;"
            " '{}' is both an input and an output".format(
                next(input_names.intersection(output_names))))

    params = inspect.signature(program).parameters

    annotations = {}
    for name, param in params.items():
        if name in input_names:
            annotations[name] = in_desc_with_name(node, state, sdfg, name)
        elif name in output_names:
            annotations[name] = out_desc_with_name(node, state, sdfg, name)
        else:
            raise ValueError(
                "'{}' was not found as an input or output for {}".format(
                    name, node.schema.name))

    program.__annotations__ = annotations

    result = DaceProgram(program, (), {})

    return result


@autoregister_params(op="Conv", name="naive_fpga")
class FPGAConv2D(ONNXForward):
    """
    The "trivial" convolution implementation, i.e. two nested maps.
    Does not work in hardware...needs some work on the unrolling etc. et.c
    """
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:

        X = in_desc_with_name(node, state, sdfg, "X")
        W = in_desc_with_name(node, state, sdfg, "W")
        try:
            B = in_desc_with_name(node, state, sdfg, "B")
        except Exception as e:
            B = None

        image_dims = len(X.shape) - 2
        num_filters = W.shape[0]
        num_channels = X.shape[1]

        if (X.dtype not in [dace.float16, dace.float32, dace.float64]
                or W.dtype not in [dace.float16, dace.float32, dace.float64]):
            return False

        # only do 2D for now
        if len(X.shape) != 4 or len(W.shape) != 4:
            return False

        if node.group != 1:
            return False

        if num_channels != W.shape[1]:
            return False

        if node.dilations is not None and (not all(d == 1
                                                   for d in node.dilations) or
                                           len(node.dilations) != image_dims):
            return False

        if node.pads is not None and (not all(p == 0 for p in node.pads)
                                      or len(node.pads) != image_dims * 2):
            return False

        if node.strides is not None and len(node.strides) != image_dims:
            return False

        if B is not None and B.shape[0] != num_filters:
            return False

        if node.auto_pad != 'NOTSET':
            return False

        return True

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:
        X = in_desc_with_name(node, state, sdfg, "X")
        W = in_desc_with_name(node, state, sdfg, "W")
        Y = out_desc_with_name(node, state, sdfg, "Y")
        try:
            B = in_desc_with_name(node, state, sdfg, "B")
        except Exception as e:
            B = None
        image_dims = len(X.shape) - 2
        strides = node.strides if node.strides is not None else [
            1 for _ in range(image_dims)
        ]
        stride_x, stride_y = strides

        if node.kernel_shape is not None:
            filter_hx, filter_hy = node.kernel_shape
        else:
            filter_hx, filter_hy = W.shape[2:]

        num_filters = W.shape[0]
        num_channels = X.shape[1]
        batch_size = X.shape[0]

        output_size_y, output_size_x = Y.shape[2:]

        new_sdfg = dace.SDFG("fpga_conv")

        new_state = new_sdfg.add_state("compute")
        new_sdfg.add_datadesc("X", copy.deepcopy(X))
        new_sdfg.add_datadesc("W", copy.deepcopy(W))
        new_sdfg.add_datadesc("Y", copy.deepcopy(Y))
        if B is not None:
            new_sdfg.add_datadesc("B", copy.deepcopy(B))
            new_sdfg.arrays["B"].transient = False

        #TODO: stride
        assert (stride_x == 1 and stride_y == 1)

        # add local storage for weights
        new_sdfg.add_array('local_W',
                           shape=W.shape,
                           dtype=W.dtype,
                           storage=dace.dtypes.StorageType.FPGA_Local,
                           transient=True)

        # add local storage for X and Y, to increase reuse

        # for X we will reuse the data of a given input channel to update the result for all output channels
        new_sdfg.add_array('local_X',
                           shape=[num_channels, filter_hx, filter_hy],
                           dtype=X.dtype,
                           storage=dace.dtypes.StorageType.FPGA_Local,
                           transient=True)

        # for Y we will reuse by accumulating on the same output channel
        new_sdfg.add_array('local_Y',
                           shape=[num_filters],
                           dtype=Y.dtype,
                           storage=dace.dtypes.StorageType.FPGA_Local,
                           transient=True)

        new_sdfg.arrays["X"].transient = False
        new_sdfg.arrays["W"].transient = False
        new_sdfg.arrays["Y"].transient = False

        # we don't need init state for Y. This is done on the fly in the tasklet

        # preload weights
        preload_W_map_entry, preload_W_map_exit = new_state.add_map(
            'preload_weights_map',
            dict(m='0:{}'.format(num_filters),
                 cin="0:{}".format(num_channels),
                 hx="0:{}".format(filter_hx),
                 hy="0:{}".format(filter_hy)))
        preload_W_task = new_state.add_tasklet("preload_weights_tasklet",
                                               inputs={"w_in"},
                                               outputs={"w_out"},
                                               code="w_out = w_in")
        # add edges
        preload_W_read = new_state.add_read("W")
        local_W_access = new_state.add_access("local_W")

        new_state.add_memlet_path(
            preload_W_read,
            preload_W_map_entry,
            preload_W_task,
            dst_conn='w_in',
            memlet=dace.Memlet(f"{preload_W_read.data}[m, cin, hx, hy]"))
        new_state.add_memlet_path(
            preload_W_task,
            preload_W_map_exit,
            local_W_access,
            src_conn='w_out',
            memlet=dace.Memlet(f"{local_W_access.data}[m, cin,hx,hy]"))

        # In pure implementation we have two maps:
        # - the outer map loops over every entry in the output array
        # - the inner inner map computes the value for a single entry in the output array (i.e. Y[b, m, x, y])

        # Here we want to increase reuse of the input feature, that is read the input once and oupdate all the
        # m output channels. Therefore we interchange some of maps indices.
        # - the outer map loops over every entry in the ouput array, not considering the channel (Y[b,:,x,y])
        # - a mid map over the input channels (this is splitted from the inner map just to have more control on unrolling)
        # - the inner computes the value for all the entries of a given point

        # the outer map loops over every entry in the output array
        outer_me, outer_mx = new_state.add_map(
            'outer_conv_map',
            dict(b="0:{}".format(batch_size),
                 out_x="0:{}".format(output_size_x),
                 out_y="0:{}".format(output_size_y)))

        mid_me, mid_mx = new_state.add_map(
            'mid_conv_map', dict(cin="0:{}".format(num_channels)))

        # the inner map computes the value for a single entry in the output array (i.e. Y[b, m, x, y])
        inner_me, inner_mx = new_state.add_map(
            'inner_conv_map',
            dict(m="0:{}".format(num_filters),
                 hx="0:{}".format(filter_hx),
                 hy="0:{}".format(filter_hy)),
            unroll=True)

        # we have to fill local_x properly: this should happen between the outer and the innermost map
        # The actual loading into local_X will be done in the tasklet, where we can add `if` conditions
        # Note: this is not pure SDFG API: the cleanest solution would involve creating another nested SDFG
        local_X_read = new_state.add_access("local_X")

        # empty memlet to create the storage
        new_state.add_memlet_path(outer_me, local_X_read, memlet=dace.Memlet())

        # Similarly, we will use local_Y to accumulate while computing in the innermost map
        local_Y_read = new_state.add_access("local_Y")
        local_Y_write = new_state.add_write("local_Y")
        new_state.add_memlet_path(outer_me, local_Y_read, memlet=dace.Memlet())

        inputs = {"image_in", "local_X_in", "filter_in", "local_Y_in"}
        if B is not None:
            inputs.add("B_in")

        # In the tasklet we read local_X (for every given input channel) and
        # we write the final result if we are computing over the last input channel
        compute_tasklet = new_state.add_tasklet(
            "compute_entry",
            inputs=inputs,
            outputs={"output", "local_Y_out"},
            code="if m==0: local_X_in = image_in\n"
            "local_Y_out = (0 if hx == 0 and hy==0 and cin==0 else local_Y_in)  + local_X_in * filter_in\n"
            # "local_X_out = local_X_in\n"
            "if hx == {}-1 and hy == {}-1 and cin=={}-1: output = local_Y_out {}"
            .format(filter_hx, filter_hy, num_channels,
                    "+ B_in" if B is not None else ""))

        filter_memlet = dace.Memlet("local_W[m, cin, hx, hy]")

        x_idx = _2d_sliding_window_index_expr(x_or_y="x",
                                              stride=stride_x,
                                              kernel_size=filter_hx)
        y_idx = _2d_sliding_window_index_expr(x_or_y="y",
                                              stride=stride_y,
                                              kernel_size=filter_hy)

        image_memlet = dace.Memlet("X[b, cin, {}, {}]".format(x_idx, y_idx))
        # hook up the inner map to the tasklet

        # local X goes inside the tasklet. Being a dynamic element, this will be codegenerated as a pointer
        # and therefore will also write back into the tile of X
        new_state.add_memlet_path(local_X_read,
                                  mid_me,
                                  inner_me,
                                  compute_tasklet,
                                  dst_conn='local_X_in',
                                  memlet=dace.Memlet(
                                      f"{local_X_read.data}[cin, hx, hy]",
                                      dynamic=True))

        # similarly, local Y
        new_state.add_memlet_path(
            local_Y_read,
            mid_me,
            inner_me,
            compute_tasklet,
            dst_conn='local_Y_in',
            memlet=dace.Memlet(f"{local_Y_read.data}[m]"))
        new_state.add_memlet_path(
            compute_tasklet,
            inner_mx,
            mid_mx,
            local_Y_write,
            src_conn='local_Y_out',
            memlet=dace.Memlet(f"{local_Y_write.data}[m]"))

        # hook up filter
        # new_state.add_edge(inner_me, None, compute_tasklet, "filter_in",
        #                    filter_memlet)
        # inner_filter_memlet = propagation.propagate_memlet(
        #     new_state, filter_memlet, inner_me, False)
        # outer_filter_memlet = propagation.propagate_memlet(
        #     new_state, inner_filter_memlet, outer_me, False)
        # new_state.add_edge(outer_me, None, inner_me, None, inner_filter_memlet)
        # new_state.add_edge(local_W_access, None, outer_me, None, outer_filter_memlet)
        new_state.add_memlet_path(local_W_access,
                                  outer_me,
                                  mid_me,
                                  inner_me,
                                  compute_tasklet,
                                  dst_conn='filter_in',
                                  memlet=filter_memlet)

        # hook up X: this goes directly to the tasklet
        read_X = new_state.add_read("X")
        # new_state.add_edge(inner_me, None, compute_tasklet, "image_in",
        #                    image_memlet)
        # inner_image_memlet = propagation.propagate_memlet(
        #     new_state, image_memlet, inner_me, False)
        # outer_image_memlet = propagation.propagate_memlet(
        #     new_state, inner_image_memlet, outer_me, False)
        # new_state.add_edge(outer_me, None, inner_me, None, inner_image_memlet)
        # new_state.add_edge(read_X, None, outer_me, None, outer_image_memlet)
        new_state.add_memlet_path(read_X,
                                  outer_me,
                                  mid_me,
                                  inner_me,
                                  compute_tasklet,
                                  dst_conn='image_in',
                                  memlet=image_memlet)

        # hook up outputs
        # The output memlet is set to be dynamic, so that the value is only written at the end of the computation
        output_memlet = dace.Memlet("Y[b, m, out_x, out_y]", dynamic=True)
        write_Y = new_state.add_write("Y")
        # inner_output_memlet = propagation.propagate_memlet(
        #     new_state, output_memlet, inner_me, False)
        # outer_output_memlet = propagation.propagate_memlet(
        #     new_state, inner_output_memlet, outer_me, False)
        # new_state.add_edge(compute_tasklet, "output", inner_mx, None,
        #                    output_memlet)
        #
        # new_state.add_edge_pair(outer_mx, inner_mx, write_Y,
        #                         inner_output_memlet, outer_output_memlet)

        new_state.add_memlet_path(compute_tasklet,
                                  inner_mx,
                                  mid_mx,
                                  outer_mx,
                                  write_Y,
                                  src_conn='output',
                                  memlet=output_memlet)

        # hook up B if required
        if B is not None:
            read_B = new_state.add_read("B")
            B_memlet = dace.Memlet("B[m]")
            new_state.add_memlet_path(read_B,
                                      outer_me,
                                      mid_me,
                                      inner_me,
                                      compute_tasklet,
                                      dst_conn='B_in',
                                      memlet=B_memlet)

        new_sdfg.fill_scope_connectors()
        new_sdfg.save('/tmp/conv.sdfg')
        return new_sdfg


@autoregister_params(op="Conv", name="fpga")
class FPGAIm2ColConv(ONNXForward):
    """ Conv implementation based on Gemm

    """
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        X = in_desc_with_name(node, state, sdfg, "X")
        W = in_desc_with_name(node, state, sdfg, "W")
        Y = out_desc_with_name(node, state, sdfg, "Y")

        try:
            B = in_desc_with_name(node, state, sdfg, "B")
        except Exception as e:
            B = None

        image_dims = len(X.shape) - 2
        num_filters = W.shape[0]
        num_channels = X.shape[1]

        # only do 2D for now
        if len(X.shape) != 4 or len(W.shape) != 4:
            return False

        if node.group != 1:
            return False

        if num_channels != W.shape[1]:
            return False

        if node.dilations is not None and (not all(d == 1
                                                   for d in node.dilations) or
                                           len(node.dilations) != image_dims):
            return False

        if node.pads is not None and (not all(p == 0 for p in node.pads)
                                      or len(node.pads) != image_dims * 2):
            return False

        if node.strides is not None and len(node.strides) != image_dims:
            return False

        if B is not None and B.shape[0] != num_filters:
            return False

        if node.auto_pad != 'NOTSET':
            return False

        # Input veclen must be equal to the output veclen
        # if X.veclen != Y.veclen:
        #     return False

        return True

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:

        X = in_desc_with_name(node, state, sdfg, "X")
        W = in_desc_with_name(node, state, sdfg, "W")
        Y = out_desc_with_name(node, state, sdfg, "Y")

        # TODO: try to vectorize input
        # Use the vector on the Y

        #TODO deal with streams

        try:
            B = in_desc_with_name(node, state, sdfg, "B")
        except Exception as e:
            B = None

        image_dims = len(X.shape) - 2
        strides = node.strides if node.strides is not None else [
            1 for _ in range(image_dims)
        ]

        if node.kernel_shape is not None:
            filter_hx, filter_hy = node.kernel_shape
        else:
            filter_hx, filter_hy = W.shape[2:]

        num_filters = W.shape[0]
        num_channels = X.shape[1]
        batch_size = X.shape[0]

        # Take output size: note, tat this accounts for vectorization (if present)
        output_size_x, output_size_y = Y.shape[2:]
        new_sdfg = dace.SDFG("fpga_im2col_conv")

        # setup inputs and outputs
        new_state = new_sdfg.add_state()
        new_sdfg.add_datadesc("X", copy.deepcopy(X))

        new_sdfg.add_datadesc("W", copy.deepcopy(W))
        new_sdfg.add_datadesc("Y", copy.deepcopy(Y))
        if B is not None:
            new_sdfg.add_datadesc("B", copy.deepcopy(B))
            new_sdfg.arrays["B"].transient = False

        new_sdfg.arrays["X"].transient = False
        new_sdfg.arrays["W"].transient = False
        new_sdfg.arrays["Y"].transient = False

        # GEMM Parameters
        vec_width = Y.veclen

        # TODO: accept parametric?

        #if Y.veclen !=1 else math.gcd(16, output_size_x)
        #N = num_filters

        K = num_channels * filter_hx * filter_hy
        M = output_size_y * output_size_x  # note that this accounts also for vectorized data types
        P = num_filters  # Num PEs  #TODO parametric
        def make_read_W(state):
            # this will read the weights, organized as a matrix of size
            # num_filters x (num_channels * filter_hx * filter_hy)

            # The original weight matrix has shape [num_filters, num_channels, filter_hx, filter_hy]

            # TODO: vectorize also this, by reading more than one element at a time, to be memory friendly
            entry, exit = state.add_map(
                "read_weights",
                {
                    "b": "0:{}".format(
                        batch_size
                    ),  # the batch map loops over every image in the batch
                    "n0": "0:{}/{}".format(num_filters, P),
                    "cin": "0:{}".format(num_channels),
                    "hx": "0:{}".format(filter_hx),
                    "hy": "0:{}".format(filter_hy),
                    "n1": "0:{}".format(P)
                },
                schedule=dace.ScheduleType.FPGA_Device)

            mem = state.add_read("W")
            pipe = state.add_write("W_pipe")
            tasklet = state.add_tasklet("read_W", {"from_memory"},
                                        {"to_kernel"},
                                        "to_kernel = from_memory")

            state.add_memlet_path(
                mem,
                entry,
                tasklet,
                dst_conn="from_memory",
                memlet=dace.Memlet("W[n0 * {} + n1, cin, hx, hy]".format(P)))
            state.add_memlet_path(tasklet,
                                  exit,
                                  pipe,
                                  src_conn="to_kernel",
                                  memlet=dace.Memlet("W_pipe[{} - n1 -1]".format(P)))

        def make_read_im2col(state, sdfg, vec_width=1):

            # Matrix B will be the im2col matrix. We will build it row-by-row
            # to facilitate streaming in the systolic GEMM, avoiding storing it back to memory
            # Note: this will require to load multiple times the input feature, yet this save I/Os
            # The im2col matrix has size (num_channels * filter_hx * filter_hy) x (output_size_y * output_size_x)

            # gear boxing: we read plain data types, we stream vector data types
            # Therefore we have two maps, the innermost is unrolled
            im2col_me, im2col_mx = state.add_map(
                "im2col_map",
                {
                    "b": "0:{}".format(batch_size),
                    "n": "0:{}/{}".format(
                        num_filters, P),  # repeat B for computing the result
                    "cin": "0:{}".format(num_channels),
                    "hx": "0:{}".format(filter_hx),
                    "hy": "0:{}".format(filter_hy),
                    "x": "0:{}".format(output_size_x),
                    "y0": "0:{}/{}".format(output_size_x,
                                           vec_width),  #TODO vectorize read
                },
                schedule=dace.ScheduleType.FPGA_Device)

            read_map_entry, read_map_exit = state.add_map(
                "unrolled_reads_X", {"y1": "0:{}".format(vec_width)},
                schedule=dace.ScheduleType.FPGA_Device,
                unroll=True)

            # local storage to accumulate data
            sdfg.add_array('vec_data_im2col',
                           shape=[vec_width],
                           dtype=dace.float32,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Registers)

            X = state.add_read("X")
            pipe = state.add_write("im2col_pipe")
            vect_data = state.add_access("vec_data_im2col")
            tasklet = state.add_tasklet("read_X", {"from_memory"},
                                        {"to_kernel"},
                                        "to_kernel = from_memory")

            im2col_input_memlet = dace.Memlet(
                "X[b, cin, x + hx, y0*{}+y1 + hy]".format(vec_width))

            # TODO check that offset to X are right in the codegenerated code

            # In the innermost map we read W=vec_width data elements and we store them into `vec_data`
            state.add_memlet_path(X,
                                  im2col_me,
                                  read_map_entry,
                                  tasklet,
                                  dst_conn="from_memory",
                                  memlet=im2col_input_memlet)

            state.add_memlet_path(tasklet,
                                  read_map_exit,
                                  vect_data,
                                  src_conn="to_kernel",
                                  memlet=dace.Memlet("vec_data_im2col[y1]"))

            # then we transfer them to the output stream
            copy_out_tasklet = state.add_tasklet('pack_and_copy_to_stream_B',
                                                 {'in_con'}, {'out_con'},
                                                 'out_con = in_con')
            state.add_memlet_path(vect_data,
                                  copy_out_tasklet,
                                  dst_conn="in_con",
                                  memlet=dace.Memlet("vec_data_im2col"))

            state.add_memlet_path(copy_out_tasklet,
                                  im2col_mx,
                                  pipe,
                                  src_conn="out_con",
                                  memlet=dace.Memlet("im2col_pipe[0]"))

        def make_write_Y(state, sdfg, vec_width, add_bias=True):

            # The resulting matrix will have size num_filter x (output_size_x, output_size_y)
            # Given the current systolic implementation, we will receive it one row at a time

            # We don't need to accumulate on Y, but we need to add Biases (if present)

            # C data arrives as expressed in vect. data type. Needs to be unpacked
            # For doing so we first store it into a local buffer and then we write it in memory
            # as gear boxing works on local data only (not global memory)

            pipe = state.add_read("Y_pipe")
            mem = state.add_write("Y")
            if add_bias is True:
                B = state.add_read("B")
            entry_map, exit_map = state.add_map(
                "write_Y", {
                    "b": "0:{}".format(batch_size),
                    "n": "0:{}".format(num_filters),
                    "x": "0:{}".format(output_size_x),
                    "y": "0:{}".format(output_size_y)
                },
                schedule=dace.ScheduleType.FPGA_Device)

            # TODO: Xilinx: do we need to unroll bias addition?

            input_connectors = {"in_con"}
            if add_bias is True: input_connectors.add("bias")
            copy__add_bias__tasklet = state.add_tasklet(
                'copy_from_stream_Y', input_connectors, {'out_con'},
                'out_con = in_con {}'.format(
                    "+ bias" if add_bias is True else ""))

            state.add_memlet_path(pipe,
                                  entry_map,
                                  copy__add_bias__tasklet,
                                  dst_conn="in_con",
                                  memlet=dace.Memlet("Y_pipe[{}-1]".format(P)))

            if add_bias is True:
                state.add_memlet_path(B,
                                      entry_map,
                                      copy__add_bias__tasklet,
                                      dst_conn="bias",
                                      memlet=dace.Memlet("B[n]"))

            # Memlet to memory

            state.add_memlet_path(copy__add_bias__tasklet,
                                  exit_map,
                                  mem,
                                  src_conn="out_con",
                                  memlet=dace.Memlet("Y[b, n, x, y]"))

        def make_compute(sdfg, state, vec_width=1):
            vec_type = dace.vector(dace.float32, vec_width)
            W_pipe_in = state.add_read("W_pipe")
            W_pipe_out = state.add_write("W_pipe")
            im2col_pipe_in = state.add_read("im2col_pipe")
            im2col_pipe_out = state.add_write("im2col_pipe")
            Y_pipe_in = state.add_read("Y_pipe")
            Y_pipe_out = state.add_write("Y_pipe")

            L = 8

            # batch_entry, batch_exit = state.add_map(
            #     "batch",  {"b": "0:{}".format(batch_size)},
            #     schedule=dace.ScheduleType.FPGA_Device)

            # We create a single flatteend pipeline
            # - we have tiling across Y: every PE computes a given number of row of the result
            # - we will drain the result for iamge i, while we compute the results of image i+1.
            #   The entire draining takes P * M clock cycles
            # - the last results are drained with an ad-hoc drain phase
            # The feeding of A is done in the first P cycle of the innermost map
            entry_pipeline, exit_pipeline = state.add_pipeline(
                "compute_and_drain",
                {
                    "b": "0:{}".format(batch_size),
                    "n0": "0:{}/{}".format(num_filters, P),
                    "k": "0:{}".format(K),
                    "m": "0:{} + {}".format(
                        M, L
                    )  # The +P is needed for the feeding: can it be eliminated?
                },
                drain_size=P * M,
                drain_overlap=False,
                additional_variables={'m_drain': 0, 'k_drain': 0},
                schedule=dace.ScheduleType.FPGA_Device)
            # entry_n0, exit_n0 = state.add_map(
            #     "batch_n0", {
            #         "b": "0:{}".format(batch_size),
            #         "n0": "0:{}/{}".format(num_filters, P),
            #     },
            #     schedule=dace.ScheduleType.FPGA_Device)
            # entry_k, exit_k = state.add_map(
            #     "k", {"k": "0:{}".format(K)},
            #     schedule=dace.ScheduleType.FPGA_Device)
            # entry_w, exit_w = state.add_map(
            #     "buffer_W", {"n1": "0:{}".format(P)},
            #     schedule=dace.ScheduleType.FPGA_Device)
            #
            # # As we are using vectorized data types for im2col, we have to consider it into these
            # # two maps
            # entry_m, exit_m = state.add_map(
            #     "m", {"m": "0:{}".format(M)},
            #     schedule=dace.ScheduleType.FPGA_Device)
            # entry_y, exit_y = state.add_map(
            #     "write_Y", {
            #         "n1": "0:{}".format(P),
            #         "m": "0:{}".format(M)
            #     },
            #     schedule=dace.ScheduleType.FPGA_Device)

            # Instantiate buffers
            sdfg.add_scalar("W_reg",
                            dtype=dace.float32,
                            transient=True,
                            storage=dace.dtypes.StorageType.FPGA_Registers)
            # This one is used for the feeding
            # sdfg.add_array("W_buf",
            #                shape=[1],
            #                dtype=dace.float32,
            #                transient=True,
            #                storage=dace.dtypes.StorageType.FPGA_Registers)
            W_reg = state.add_write("W_reg")
            # W_buf = state.add_write("W_buf")

            sdfg.add_scalar("fake_dep",
                            dtype=dace.int32,
                            transient=True,
                            storage=dace.dtypes.StorageType.FPGA_Registers)
            fake_dep = state.add_access("fake_dep")
            # For Y result we are going to use vectorized data type
            sdfg.add_array(
                "Y_buffer",
                [M],  #M already accounts for vec width
                dtype=vec_type,
                transient=True,
                storage=dace.dtypes.StorageType.FPGA_Local)
            sdfg.add_array("Y_reg",
                           shape=[1],
                           dtype=vec_type,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Registers)
            Y_buffer_in = state.add_read("Y_buffer")
            Y_buffer_out = state.add_write("Y_buffer")

            # FEED W
            # every PE: reads input data in the first P cycles of the innermost loop,
            # buffers the data assigned to it, forwards the data
#             read_w_tasklet = state.add_tasklet(
#                 "read_w", {"w_in"}, {"w_buf"}, """\
# if m < {} and  not {}:
#     w_buf = w_in""".format(P, entry_pipeline.pipeline.drain_condition()))

            read_w_tasklet = state.add_tasklet(
                "buffer_w", {"w_in"}, {"w_reg"}, """\
if m == 0 and not {}:
    w_reg = w_in""".format(entry_pipeline.pipeline.drain_condition()))

            # Memlet to the conditional feed tasklet. Notice that these are dynamic to
            # perform reads/write to steams only when really needed
            state.add_memlet_path(W_pipe_in,
                                  entry_pipeline,
                                  read_w_tasklet,
                                  memlet=dace.Memlet("W_pipe[p]",
                                                     dynamic=True),
                                  dst_conn="w_in")
            # state.add_memlet_path(read_w_tasklet,
            #                       W_buf,
            #                       memlet=dace.Memlet("W_buf[0]", dynamic=True),
            #                       src_conn="w_buf")
            # state.add_memlet_path(W_buf,
            #                       buffer_and_forward_w_tasklet,
            #                       memlet=dace.Memlet("W_buf[0]", dynamic=True),
            #                       dst_conn="w_buf")
            # state.add_memlet_path(buffer_and_forward_w_tasklet,
            #                       exit_pipeline,
            #                       W_pipe_out,
            #                       memlet=dace.Memlet("W_pipe[p + 1]",
            #                                          dynamic=True),
            #                       src_conn="w_out")
            state.add_memlet_path(read_w_tasklet,
                                  W_reg,
                                  memlet=dace.Memlet("W_reg[0]", dynamic=True),
                                  src_conn="w_reg")

            # FEED B (im2col matrix)
            # Read B: done outside of the compute tasklet to help type inference
            sdfg.add_array("im2col_reg",
                           shape=[1],
                           dtype=vec_type,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Local)
            im2col_reg = state.add_access("im2col_reg")
            buffer_im2col_tasklet = state.add_tasklet(
                "buffer_im2col", {"im2col_in"}, {"im2col_reg"}, """\
if m>={} and not {}:
    im2col_reg = im2col_in""".format(L, entry_pipeline.pipeline.drain_condition()))

            state.add_memlet_path(im2col_pipe_in,
                                  entry_pipeline,
                                  buffer_im2col_tasklet,
                                  memlet=dace.Memlet("im2col_pipe[p]",
                                                     dynamic=True),
                                  dst_conn="im2col_in")
            state.add_memlet_path(buffer_im2col_tasklet,
                                  im2col_reg,
                                  memlet=dace.Memlet("im2col_reg[0]",
                                                     dynamic=True),
                                  src_conn="im2col_reg")

            # DRAIN: attention, this must be  theoretically done before starting to compute the result for the next tile
            # with this implementation is still done after: however, since for the first P cycle we don't overwrite Y_buffer
            # this is still safe
            # Condition for draining:
            # - we completed one of the assigned image and we are working on the first assigned row of the next (b>0 and n0==0)
            # - or, we are not working on the first assigned row (n0>0)
            # - we have data to drain (k<P && m<M. Notice tha k identifies the PE that is actually draining)
            # - or we are in drain phase of the pipeline (draining the last tile)
            # Notice that the initial P iteration over P are devoted to feed the data

            # Hack: we have to add explicitly the increase of m and k while in the draining phase,
            # as this is not done automatically by the pipeline scope
            write_y_tasklet = state.add_tasklet(
                "write_y", {"buffer_in", "forward_in"}, {"y_pipe_out", "fake_dep_out"}, f"""\
if ((b>0  or n0 > 0)  and k_drain <=p and m_drain <{M})  or {entry_pipeline.pipeline.drain_condition()}:
    y_pipe_out = forward_in if p > 0 and k_drain > 0 else buffer_in
if not {entry_pipeline.pipeline.drain_condition()}:\n\t
    if m_drain >=  {L} + {M} -1:
        m_drain = 0
        if k_drain >= {K} - 1:
            k_drain = 0
        else:
            k_drain = k_drain +1
    else:
        m_drain = m_drain + 1
else:
    if m_drain >=  {M} -1:
        m_drain = 0
        k_drain = k_drain + 1
    else:
        m_drain = m_drain + 1
fake_dep_out=0
    """
)
            # add allow oob for this memlet
            state.add_memlet_path(Y_buffer_in,
                                  entry_pipeline,
                                  write_y_tasklet,
                                  memlet=dace.Memlet("Y_buffer[m_drain]",
                                                     dynamic=True, allow_oob=True),
                                  dst_conn="buffer_in")
            state.add_memlet_path(Y_pipe_in,
                                  entry_pipeline,
                                  write_y_tasklet,
                                  memlet=dace.Memlet("Y_pipe[p-1]",
                                                     dynamic=True),
                                  dst_conn="forward_in")
            state.add_memlet_path(write_y_tasklet,
                                  exit_pipeline,
                                  Y_pipe_out,
                                  memlet=dace.Memlet("Y_pipe[p]",
                                                     dynamic=True),
                                  src_conn="y_pipe_out")

            # COMPUTE
            # Compute and forward B: this is done if we are not in the init phase of the pipeline
            compute_tasklet = state.add_tasklet(
                "multiply_add", {"w_in", "im2col_in", "y_in", "fake_dep_in"},
                {"im2col_out", "y_out"}, """\
if m>= {} and not {}:
    y_prev = 0 if k == 0 else y_in 
    y_out = y_prev + w_in * im2col_in
    if p < {} - 1:
        im2col_out = im2col_in""".format(L, entry_pipeline.pipeline.drain_condition(), P))

            state.add_memlet_path(W_reg,
                                  compute_tasklet,
                                  dst_conn="w_in",
                                  memlet=dace.Memlet("W_reg[0]"))
            # B to/from compute tasklet
            state.add_memlet_path(im2col_reg,
                                  compute_tasklet,
                                  memlet=dace.Memlet("im2col_reg[0]",
                                                     dynamic=True),
                                  dst_conn="im2col_in")
            state.add_memlet_path(compute_tasklet,
                                  exit_pipeline,
                                  im2col_pipe_out,
                                  memlet=dace.Memlet("im2col_pipe[p + 1]",
                                                     dynamic=True),
                                  src_conn="im2col_out")
            Y_buffer_to_compute_y_in = dace.Memlet("Y_buffer[m-{}]".format(L))
            Y_buffer_to_compute_y_in.allow_oob = True
            state.add_memlet_path(Y_buffer_in,
                                  entry_pipeline,
                                  compute_tasklet,
                                  dst_conn="y_in",
                                  memlet=Y_buffer_to_compute_y_in)
            state.add_memlet_path(
                compute_tasklet,
                Y_buffer_out,
                memlet=dace.Memlet("Y_buffer[m-{}]".format(L), dynamic=True),
                src_conn="y_out")
            state.add_memlet_path(Y_buffer_out,
                                  exit_pipeline,
                                  memlet=dace.Memlet())

            #             # Compute and forward B
            #             compute_tasklet = state.add_tasklet(
            #                 "multiply_add", {"w_in", "im2col_in", "y_in"},
            #                 {"im2col_out", "y_out"}, """\
            # y_prev = 0 if k == 0 else y_in
            # y_out = y_prev + w_in * im2col_in
            # if p < {P} - 1:
            #     im2col_out = im2col_in""".format(P=P))
            #
            #             state.add_memlet_path(W_reg,
            #                                   entry_m,
            #                                   compute_tasklet,
            #                                   dst_conn="w_in",
            #                                   memlet=dace.Memlet("W_reg[0]"))
            #             state.add_memlet_path(im2col_pipe_in,
            #                                   entry_n0,
            #                                   entry_k,
            #                                   entry_m,
            #                                   compute_tasklet,
            #                                   memlet=dace.Memlet("im2col_pipe[p]",
            #                                                      dynamic=False),
            #                                   dst_conn="im2col_in")
            #             state.add_memlet_path(compute_tasklet,
            #                                   exit_m,
            #                                   exit_k,
            #                                   exit_n0,
            #                                   im2col_pipe_out,
            #                                   memlet=dace.Memlet("im2col_pipe[p + 1]",
            #                                                      dynamic=True),
            #                                   src_conn="im2col_out")
            #             state.add_memlet_path(Y_buffer_in,
            #                                   entry_k,
            #                                   entry_m,
            #                                   compute_tasklet,
            #                                   dst_conn="y_in",
            #                                   memlet=dace.Memlet("Y_buffer[m]"))
            #             state.add_memlet_path(entry_n0, Y_buffer_in, memlet=dace.Memlet())
            #             state.add_memlet_path(compute_tasklet,
            #                                   exit_m,
            #                                   exit_k,
            #                                   Y_buffer_out,
            #                                   src_conn="y_out",
            #                                   memlet=dace.Memlet("Y_buffer[m]"))
            #             state.add_memlet_path(Y_buffer_out, exit_n0, memlet=dace.Memlet())
            # DRAIN
            #             write_y_tasklet = state.add_tasklet(
            #                 "write_y", {"buffer_in", "forward_in"}, {"y_out"}, """\
            # if n1 <= p:
            #     y_out = forward_in if p > 0 and n1 > 0 else buffer_in""")
            #             state.add_memlet_path(Y_buffer_out,
            #                                   entry_y,
            #                                   write_y_tasklet,
            #                                   memlet=dace.Memlet("Y_buffer[m]",
            #                                                      dynamic=True),
            #                                   dst_conn="buffer_in")
            #             state.add_memlet_path(Y_pipe_in,
            #                                   entry_n0,
            #                                   entry_y,
            #                                   write_y_tasklet,
            #                                   memlet=dace.Memlet("Y_pipe[p-1]",
            #                                                      dynamic=True),
            #                                   dst_conn="forward_in")
            #             state.add_memlet_path(write_y_tasklet,
            #                                   exit_y,
            #                                   exit_n0,
            #                                   Y_pipe_out,
            #                                   src_conn="y_out",
            #                                   memlet=dace.Memlet("Y_pipe[p]",
            #                                                      dynamic=True))

            # Unroll processing elements
            compute_entry, compute_exit = state.add_map(
                "unroll_compute", {"p": "0:{}".format(P)},
                schedule=dace.ScheduleType.FPGA_Device,
                unroll=True)

            # Bring data nodes into scope
            state.add_memlet_path(compute_entry,
                                  W_pipe_in,
                                  memlet=dace.memlet.Memlet())
            state.add_memlet_path(compute_entry,
                                  im2col_pipe_in,
                                  memlet=dace.memlet.Memlet())
            state.add_memlet_path(compute_entry,
                                  Y_pipe_in,
                                  memlet=dace.memlet.Memlet())
            state.add_memlet_path(W_pipe_out,
                                  compute_exit,
                                  memlet=dace.memlet.Memlet())
            state.add_memlet_path(im2col_pipe_out,
                                  compute_exit,
                                  memlet=dace.memlet.Memlet())
            state.add_memlet_path(Y_pipe_out,
                                  compute_exit,
                                  memlet=dace.memlet.Memlet())
            state.add_memlet_path(write_y_tasklet, fake_dep, src_conn="fake_dep_out",
                                  memlet=dace.memlet.Memlet("fake_dep[0]", dynamic=True))
            state.add_memlet_path(fake_dep, compute_tasklet, dst_conn="fake_dep_in",
                                  memlet=dace.memlet.Memlet("fake_dep[0]", dynamic=True))
            # Add empty memlet to define the registers at the right place
            im2col_init = state.add_access("im2col_reg")
            state.add_memlet_path(compute_entry,
                                  im2col_init,
                                  memlet=dace.Memlet())
            state.add_memlet_path(im2col_init,
                                  entry_pipeline,
                                  memlet=dace.Memlet())
            state.add_memlet_path(compute_entry,
                                  Y_buffer_in,
                                  memlet=dace.Memlet())
            W_reg_init = state.add_write("W_reg")
            state.add_memlet_path(compute_entry,
                                  W_reg_init,
                                  memlet=dace.Memlet())
            state.add_memlet_path(W_reg_init,
                                  entry_pipeline,
                                  memlet=dace.Memlet())

        # build the compute State
        vec_type = dace.vector(dace.float32, vec_width)

        new_sdfg.add_stream("W_pipe",
                            dace.float32,
                            transient=True,
                            shape=(P,),
                            storage=dace.dtypes.StorageType.FPGA_Local,
                            buffer_size=P+2)
        new_sdfg.add_stream("im2col_pipe",
                            vec_type,
                            transient=True,
                            shape=(P + 1, ),
                            buffer_size=P + 2,
                            storage=dace.dtypes.StorageType.FPGA_Local)
        new_sdfg.add_stream("Y_pipe",
                            vec_type,
                            transient=True,
                            shape=(P + 1, ),
                            buffer_size=P + 2,
                            storage=dace.dtypes.StorageType.FPGA_Local)

        make_read_W(new_state)
        make_read_im2col(new_state, new_sdfg, vec_width)
        make_compute(new_sdfg, new_state, vec_width)
        make_write_Y(new_state, new_sdfg, vec_width, add_bias=(B is not None))

        new_sdfg.fill_scope_connectors()
        # Specialize the new sdfg, by using the input shapes
        new_sdfg.save("/tmp/conv.sdfg")
        # new_sdfg.validate()
        return new_sdfg


@autoregister_params(op="Relu", name="fpga")
class FPGARelu(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        X = in_desc_with_name(node, state, sdfg, "X")
        Y = out_desc_with_name(node, state, sdfg, "Y")

        # Input veclen must be equal to the output veclen
        # if X.veclen != Y.veclen:
        #     return False
        return True

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        X = in_desc_with_name(node, state, sdfg, "X")
        Y = out_desc_with_name(node, state, sdfg, "Y")

        # TODO deal with this. Right Now I'm doing it to
        # gently introduce streaming
        vec_width = X.veclen
        # if node.name in["ONNX_Relu_1", "ONNX_Relu_3", "ONNX_Relu_9", "ONNX_Relu_11"]:
        #     streaming_node = True
        #     # Use the vector on the X
        #     print("RELU streamed ----")
        # else:
        #     streaming_node = False
        #     print("RELU NON streamed ----")
        streaming_node = False
        if X.veclen != Y.veclen:
            # we will need to copy the data out accordingly
            # NOTE: for the moment, tested with Y veclen = 1
            vec_width_mismatch = True
        else:
            vec_width_mismatch = False

        # Build map ranges: one loop per dimension
        map_ranges = {'__i%d' % i: '0:%s' % n for i, n in enumerate(X.shape)}

        new_sdfg = dace.SDFG("fpga_relu")

        new_state = new_sdfg.add_state("compute")
        new_sdfg.add_datadesc("X", copy.deepcopy(X))
        new_sdfg.add_datadesc("Y", copy.deepcopy(Y))

        new_sdfg.arrays["X"].transient = False
        new_sdfg.arrays["Y"].transient = False
        outer_me, outer_mx = new_state.add_map('relu_map', map_ranges)

        new_sdfg.add_array("vec_data_in", [vec_width],
                           dtype=dace.float32,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Registers)
        new_sdfg.add_array("vec_data_out", [1],
                           dtype=X.dtype,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Registers)

        vec_data_in = new_state.add_access("vec_data_in")
        vec_data_out = new_state.add_access("vec_data_in")

        # Unrolled map to compute the elementwise max
        inner_me, inner_mx = new_state.add_map(
            'inner_relu_map', dict(i="0:{}".format(vec_width)), unroll=True)

        # read_tasklet = new_state.add_tasklet('read_task', ['in_con'], ['out_con'],
        #                                 'out_con=in_con')
        # write_tasklet = new_state.add_tasklet('write_task', ['in_con'], ['out_con'],
        #                                      'out_con=in_con')
        tasklet = new_state.add_tasklet('relu_task', ['x_con'], ['y_con'],
                                        'y_con = max(0.0, x_con)')
        x_read = new_state.add_read("X")
        y_write = new_state.add_write("Y")

        #unpack vector data
        #memlet from memory
        if not streaming_node:
            new_state.add_memlet_path(
                x_read,
                outer_me,
                vec_data_in,
                memlet=dace.Memlet("X[{}]".format(",".join(
                    ['__i%d' % i for i in range(len(X.shape))]))))
        else:
            #memlet from stream
            new_state.add_memlet_path(x_read,
                                      outer_me,
                                      vec_data_in,
                                      memlet=dace.Memlet("X[0,0,0,0]"))

        # connect to tasklet
        new_state.add_memlet_path(vec_data_in,
                                  inner_me,
                                  tasklet,
                                  dst_conn='x_con',
                                  memlet=dace.Memlet("vec_data_in[i]"))

        # pack
        new_state.add_memlet_path(tasklet,
                                  inner_mx,
                                  vec_data_out,
                                  src_conn='y_con',
                                  memlet=dace.Memlet("vec_data_in[i]"))

        # if there is a mismatch between input and output veclen (e.g. GEMM->Relu in Lenet)
        # we need an extra loop here

        if vec_width_mismatch:
            #TODO: right now this handle the case Y.veclen==1
            assert (Y.veclen == 1)
            write_out_me, write_out_mx = new_state.add_map(
                'relu_write_out_map', dict(i="0:{}".format(vec_width)))
            tasklet = new_state.add_tasklet('read_tasklet', ['_in'], ['_out'],
                                            code="_out = _in")
            # write out
            new_state.add_memlet_path(vec_data_out,
                                      write_out_me,
                                      tasklet,
                                      dst_conn="_in",
                                      memlet=dace.Memlet("vec_data_in[i]"))
            # TODO: special case for GEMM->Relu, do the right memlet
            new_state.add_memlet_path(
                tasklet,
                write_out_mx,
                outer_mx,
                y_write,
                src_conn="_out",
                memlet=dace.Memlet("Y[__i0, __i1*{}+i]".format(vec_width)))

        else:
            #write out
            new_state.add_memlet_path(
                vec_data_out,
                outer_mx,
                y_write,
                memlet=dace.Memlet("Y[{}]".format(",".join(
                    ['__i%d' % i for i in range(len(X.shape))]))))
        new_sdfg.fill_scope_connectors()
        new_sdfg.save('/tmp/relu.sdfg')
        return new_sdfg


@autoregister_params(op="MaxPool", name="fpga")
class FPGAMaxPool2D(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        X = in_desc_with_name(node, state, sdfg, "X")

        if "Indices" in {e.src_conn for e in state.out_edges(node)}:
            return False

        image_dims = len(X.shape) - 2

        # only do 2D for now
        if image_dims != 2:
            return False

        if node.pads is not None and (not all(p == 0 for p in node.pads)
                                      or len(node.pads) != image_dims * 2):
            return False

        if node.strides is not None and len(node.strides) != image_dims:
            return False

        if node.auto_pad != 'NOTSET':
            return False

        if node.ceil_mode != 0 or node.storage_order != 0:
            return False

        if node.dilations is not None and (not all(d == 1
                                                   for d in node.dilations) or
                                           len(node.dilations) != image_dims):
            return False
        return True

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        # MAX Pool: the current implementation exploit a sliding window. Considering a single batch and a single
        # channel, we will read one input element at a time, shifting

        #TODO: this implementation depends on how data will be streamed
        # for the moment being we assume it sends one channel after the other

        # TODO: unroll reads from memory/stream
        # TODO: pay attention to do not mix height, width

        X = in_desc_with_name(node, state, sdfg, "X")
        Y = out_desc_with_name(node, state, sdfg, "Y")
        vec_width = X.veclen

        image_dims = len(X.shape) - 2
        batch_size = X.shape[0]
        num_channels = X.shape[1]
        strides = node.strides if node.strides is not None else [
            1 for _ in range(image_dims)
        ]
        stride_height, stride_width = strides
        filter_height, filter_width = node.kernel_shape
        input_size_height, input_size_width = X.shape[2:]
        output_size_y, output_size_x = Y.shape[2:]

        new_sdfg = dace.SDFG("fpga_maxpool")
        new_state = new_sdfg.add_state("compute")

        # we don't need initialization

        new_sdfg.add_datadesc("X", copy.deepcopy(X))
        new_sdfg.add_datadesc("Y", copy.deepcopy(Y))
        new_sdfg.arrays["X"].transient = False
        new_sdfg.arrays["Y"].transient = False

        #shift register. Note that this contains plain data types
        shift_register_size = input_size_width * vec_width * (
            filter_height - 1) + (filter_width - 1) + 1

        #TODO: use X dtype
        new_sdfg.add_array("shift_register", [shift_register_size],
                           dace.float32,
                           storage=dace.StorageType.FPGA_ShiftRegister,
                           transient=True)
        # variable for reduction
        new_sdfg.add_array("max_res", [1],
                           dace.float32,
                           storage=dace.StorageType.FPGA_Registers,
                           transient=True)
        new_sdfg.add_array('vec_data',
                           shape=[vec_width],
                           dtype=dace.float32,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Registers)
        # temporary storage for unpacked vector data type

        # the outer map loops over every entry in the input array
        # (useful also in the case of streaming input, we can't skip data
        # Note that `input_size_width` accounts for vectorziation
        outer_me, outer_mx = new_state.add_map(
            'outer_pool_map',
            dict(b="0:{}".format(batch_size),
                 c="0:{}".format(num_channels),
                 in_y="0:{}".format(input_size_height),
                 in_x="0:{}".format(input_size_width)))

        # if vec_width >1 this will deal with it
        vect_me, vect_mx = new_state.add_map('vect_pool_map',
                                             dict(w="0:{}".format(vec_width)))

        # the inner map computes the pooling
        inner_me, inner_mx = new_state.add_map(
            'inner_pool_map',
            dict(hy="0:{}".format(filter_height),
                 hx="0:{}".format(filter_width)),
            unroll=True)

        # read data into vec data
        # tasklet = new_state.add_tasklet('read_tasklet', ['_in'], ['_out'], code="_out = _in")

        # compute the maximum: we can compute always, but we can write the result only
        # according to the slide and at the end of the filter loops
        # NOTE: in_x could reflect the fact that it is vctorized
        compute_tasklet = new_state.add_tasklet(
            "compute_entry",
            inputs={"image_in", "max_in"},
            outputs={"output", "max_out"},
            #code="output = image_in"
            code="if hx == 0 and hy == 0: max_in = {}\n"  #init
            "max_out = float(max(max_in, image_in))\n"
            "if hy == {} - 1 and hx == {} -1 and  in_y % {} == {} - 1 and (in_x *{}+w) % {} == {} -1: output = max_out"
            .format(dtypes.min_value(Y.dtype), filter_height, filter_width,
                    filter_height, filter_height, vec_width, filter_height,
                    filter_width))

        shift_register = new_state.add_access("shift_register")

        read_X = new_state.add_read("X")
        write_Y = new_state.add_write("Y")
        read_max_res = new_state.add_access("max_res")
        write_max_res = new_state.add_write("max_res")
        vec_data = new_state.add_access("vec_data")

        # memlet: from input image to vec data
        # new_state.add_memlet_path(
        #     read_X,
        #     outer_me,
        #     tasklet,
        #     dst_conn="_in",
        #     memlet=dace.Memlet("X[b, c, in_y, in_x]"))
        # new_state.add_memlet_path(
        #     tasklet,
        #     vec_data,
        #     src_conn="_out",
        #     memlet=dace.Memlet("vec_data[0]")
        # )

        new_state.add_memlet_path(read_X,
                                  outer_me,
                                  vec_data,
                                  dst_conn="_in",
                                  memlet=dace.Memlet("X[b, c, in_y, in_x]"))

        # memlet: from input image to shift register
        to_shift_register_memlet = dace.Memlet(
            "vec_data[w]", other_subset="{}".format(shift_register_size - 1))
        # explicitely set oob otherwise is not taken
        to_shift_register_memlet.allow_oob = True
        new_state.add_memlet_path(vec_data,
                                  vect_me,
                                  shift_register,
                                  memlet=to_shift_register_memlet,
                                  propagate=False)

        # To create the shift register outside the map, add an empty memlet path
        # shift_register_write = new_state.add_write("shift_register")
        shift_register_read = new_state.add_read("shift_register")
        # new_state.add_memlet_path(shift_register_read,
        #                           outer_me,
        #                           # vect_me,
        #                           inner_me,
        #                           inner_mx,
        #                           # vect_mx,
        #                           outer_mx,
        #                           shift_register_write,
        #                           memlet=dace.Memlet())
        new_state.add_memlet_path(shift_register_read,
                                  outer_me,
                                  memlet=dace.Memlet())
        # new_state.add_memlet_path(outer_mx, shift_register_write, memlet=dace.Memlet())

        # memlet from shift register to max tasklet
        # NOTE: vec width
        new_state.add_memlet_path(shift_register,
                                  inner_me,
                                  compute_tasklet,
                                  dst_conn="image_in",
                                  memlet=dace.Memlet(
                                      "shift_register[hy*{}+hx]".format(
                                          input_size_width * vec_width)))

        #memlets for max
        new_state.add_memlet_path(read_max_res,
                                  inner_me,
                                  compute_tasklet,
                                  dst_conn="max_in",
                                  memlet=dace.Memlet("max_res[0]"))
        #empty memlet
        new_state.add_memlet_path(vect_me, read_max_res, memlet=dace.Memlet())

        new_state.add_memlet_path(compute_tasklet,
                                  inner_mx,
                                  write_max_res,
                                  src_conn="max_out",
                                  memlet=dace.Memlet("max_res[0]"))
        #empty memlet
        new_state.add_memlet_path(write_max_res, vect_mx, memlet=dace.Memlet())
        #Attention, the storing location must take into account that the input was vectorized
        y_memlet = dace.Memlet("Y[b,c, in_y//{}, (in_x*{}+w)//{}]".format(
            filter_height, vec_width, filter_width))
        #dynamic memlet (to access only when needed) from compute tasklet to out image
        # Attention: use propagate=False otherwise it does not validate
        new_state.add_memlet_path(compute_tasklet,
                                  inner_mx,
                                  vect_mx,
                                  outer_mx,
                                  write_Y,
                                  src_conn="output",
                                  memlet=y_memlet,
                                  propagate=True)

        new_sdfg.fill_scope_connectors()
        new_sdfg.save("/tmp/maxpool.sdfg")
        return new_sdfg


@autoregister_params(op="Gemm", name="fpga")
class FPGAGemm(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        if node.alpha == 1.0 and node.beta == 1.0 and node.transA == 0 and node.transB == 1:
            return True
        return False

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        node.validate(sdfg, state)

        assert node.alpha == 1.0 and node.beta == 1.0 and node.transA == 0 and node.transB == 1

        A = in_desc_with_name(node, state, sdfg, "A")
        B = in_desc_with_name(node, state, sdfg, "B")
        C = in_desc_with_name(node, state, sdfg, "C")
        Y = out_desc_with_name(node, state, sdfg, "Y")

        new_sdfg = dace.SDFG("fpga_gemm")
        new_state = new_sdfg.add_state("compute")
        new_sdfg.add_datadesc("A", copy.deepcopy(A))
        new_sdfg.add_datadesc("B", copy.deepcopy(B))
        new_sdfg.add_datadesc("C", copy.deepcopy(C))
        new_sdfg.add_datadesc("Y", copy.deepcopy(Y))
        new_sdfg.arrays["A"].transient = False
        new_sdfg.arrays["B"].transient = False
        new_sdfg.arrays["C"].transient = False
        new_sdfg.arrays["Y"].transient = False

        # GEMM Parameters

        N = A.shape[0]
        K = A.shape[1]
        # for the sake of optimization, the input C is non vectorized
        # while the output Y can be vectorized
        M_C = C.shape[0]
        M_Y = Y.shape[1]
        P = math.gcd(N, 16)  # Num PEs
        vec_width = Y.veclen
        if node.name == "ONNX_Gemm_8":
            streamed_node = True
            print("{} streamed".format(node.name))
        else:
            streamed_node = False
            print("{} non streamed".format(node.name))

        ####################################################
        # Build the SDFG: starting point: gemm_fpga_systolic vectorized sample

        def make_read_A(state):

            # TODO: vectorize also this, by reading more than one element at a time
            entry, exit = state.add_map("read_A", {
                "n0": "0:{}/{}".format(N, P),
                "k": "0:{}".format(K),
                "n1": "0:{}".format(P)
            },
                                        schedule=dace.ScheduleType.FPGA_Device)

            mem = state.add_read("A")
            pipe = state.add_write("A_pipe")
            tasklet = state.add_tasklet("read_A", {"from_memory"},
                                        {"to_kernel"},
                                        "to_kernel = from_memory")

            state.add_memlet_path(mem,
                                  entry,
                                  tasklet,
                                  dst_conn="from_memory",
                                  memlet=dace.Memlet(
                                      "A[n0 * {} + n1, k]".format(P)))
            state.add_memlet_path(tasklet,
                                  exit,
                                  pipe,
                                  src_conn="to_kernel",
                                  memlet=dace.Memlet("A_pipe[0]"))

        def make_read_B(state, sdfg, vec_width=1):

            # NOTE: We are reading this transposed: B is originally a matrix MxK

            # B is accessed by row
            # gear boxing: we read plain data types, we stream vector data types
            # Therefore we have two maps, the innermost is unrolled
            entry, exit = state.add_map("read_B", {
                "n": "0:{}/{}".format(N, P),
                "m": "0:{}".format(K),
                "k0": "0:{}/{}".format(M_C, vec_width)
            },
                                        schedule=dace.ScheduleType.FPGA_Device)

            read_map_entry, read_map_exit = state.add_map(
                "unrolled_reads_B", {"k1": "0:{}".format(vec_width)},
                schedule=dace.ScheduleType.FPGA_Device,
                unroll=True)

            # local storage to accumulate data
            sdfg.add_array('vec_data_B',
                           shape=[vec_width],
                           dtype=dace.float32,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Registers)
            mem = state.add_read("B")
            pipe = state.add_write("B_pipe")
            vect_data = state.add_access("vec_data_B")
            tasklet = state.add_tasklet("read_B", {"from_memory"},
                                        {"to_kernel"},
                                        "to_kernel = from_memory")

            # In the innermost map we read W=vec_width data elements and we store them into `vec_data`
            state.add_memlet_path(mem,
                                  entry,
                                  read_map_entry,
                                  tasklet,
                                  dst_conn="from_memory",
                                  memlet=dace.Memlet(
                                      "B[k0*{}+k1, m]".format(vec_width)))

            state.add_memlet_path(tasklet,
                                  read_map_exit,
                                  vect_data,
                                  src_conn="to_kernel",
                                  memlet=dace.Memlet("vec_data_B[k1]"))

            # then we transfer them to the output stream
            copy_out_tasklet = state.add_tasklet('pack_and_copy_to_stream_B',
                                                 {'in_con'}, {'out_con'},
                                                 'out_con = in_con')
            state.add_memlet_path(vect_data,
                                  copy_out_tasklet,
                                  dst_conn="in_con",
                                  memlet=dace.Memlet("vec_data_B"))

            state.add_memlet_path(copy_out_tasklet,
                                  exit,
                                  pipe,
                                  src_conn="out_con",
                                  memlet=dace.Memlet("B_pipe[0]"))

        def make_write_C(state, sdfg, vec_width):

            # C data arrives as expressed in vect. data type. Needs to be unpacked
            # For doing so we first store it into a local buffer and then we write it in memory
            # as gear boxing works on local data only (not global memory)

            # Terrible hack to deal with different vec size between C and Y
            if C.veclen != Y.veclen:
                deal_with_misread = True
            else:
                deal_with_misread = False

            pipe = state.add_read("C_pipe")
            mem_read = state.add_read("C")
            mem = state.add_write("Y")

            entry_map, exit_map = state.add_map(
                "write_C",
                {
                    "n": "0:{}".format(N),
                    "m": "0:{}".format(M_Y)  #consider also vectorization
                },
                schedule=dace.ScheduleType.FPGA_Device)

            #
            # # local storage to accumulate data
            # sdfg.add_array('vec_data_C',
            #                shape=[vec_width],
            #                dtype=dace.float32,
            #                transient=True,
            #                storage=dace.dtypes.StorageType.FPGA_Registers)
            #
            # vect_data = state.add_access("vec_data_C")

            # then we transfer them to the output stream
            # copy_in_tasklet = state.add_tasklet('copy_from_stream_C',
            #                                     {'in_con'}, {'out_con'},
            #                                     'out_con = in_con')

            # state.add_memlet_path(pipe,
            #                       entry_map,
            #                       copy_in_tasklet,
            #                       dst_conn="in_con",
            #                       memlet=dace.Memlet("C_pipe[{}-1]".format(P)))
            # # this will trigger gear boxing
            # state.add_memlet_path(copy_in_tasklet,
            #                       vect_data,
            #                       src_conn="out_con",
            #                       memlet=dace.Memlet("vec_data_C"))

            # then we copy that to memory

            if deal_with_misread:
                add_map_entry, add_map_exit = state.add_map(
                    "add_C", {"m1": "0:{}".format(vec_width)},
                    schedule=dace.ScheduleType.FPGA_Device,
                    unroll=True)
                # local storage to accumulate data
                sdfg.add_array('vec_data_C',
                               shape=[vec_width],
                               dtype=dace.float32,
                               transient=True,
                               storage=dace.dtypes.StorageType.FPGA_Registers)

                vect_data = state.add_access("vec_data_C")
                # local storage to accumulate data
                sdfg.add_array('vec_res',
                               shape=[vec_width],
                               dtype=dace.float32,
                               transient=True,
                               storage=dace.dtypes.StorageType.FPGA_Registers)
                vect_res = state.add_access("vec_res")

                # then we transfer them to the output stream
                copy_in_tasklet = state.add_tasklet('copy_from_stream_C',
                                                    {'in_con'}, {'out_con'},
                                                    'out_con = in_con')

                state.add_memlet_path(pipe,
                                      entry_map,
                                      copy_in_tasklet,
                                      dst_conn="in_con",
                                      memlet=dace.Memlet(
                                          "C_pipe[{}-1]".format(P)))
                # this will trigger gear boxing
                state.add_memlet_path(copy_in_tasklet,
                                      vect_data,
                                      src_conn="out_con",
                                      memlet=dace.Memlet("vec_data_C"))

                # add C
                add_C_tasklet = state.add_tasklet('add_C_tasklet',
                                                  {'in_con', 'prev_c'},
                                                  {'out_con'},
                                                  'out_con = in_con + prev_c')
                state.add_memlet_path(vect_data,
                                      add_map_entry,
                                      add_C_tasklet,
                                      dst_conn="in_con",
                                      memlet=dace.Memlet("vec_data_C[m1]"))
                state.add_memlet_path(mem_read,
                                      entry_map,
                                      add_map_entry,
                                      add_C_tasklet,
                                      dst_conn="prev_c",
                                      memlet=dace.Memlet(
                                          "C[m*{}+m1]".format(vec_width)))

                # write out
                state.add_memlet_path(add_C_tasklet,
                                      add_map_exit,
                                      vect_res,
                                      src_conn="out_con",
                                      memlet=dace.Memlet("vec_res[m1]"))
                state.add_memlet_path(vect_res,
                                      exit_map,
                                      mem,
                                      memlet=dace.Memlet("Y[n,m]"))

            else:
                tasklet = state.add_tasklet(
                    "write_C", {"from_kernel", "prev_c"}, {"to_memory"},
                    "to_memory = from_kernel + prev_c")
                state.add_memlet_path(pipe,
                                      entry_map,
                                      tasklet,
                                      dst_conn="from_kernel",
                                      memlet=dace.Memlet(
                                          "C_pipe[{}-1]".format(P)))
                state.add_memlet_path(mem_read,
                                      entry_map,
                                      tasklet,
                                      dst_conn="prev_c",
                                      memlet=dace.Memlet("C[m]"))
                state.add_memlet_path(tasklet,
                                      exit_map,
                                      mem,
                                      src_conn="to_memory",
                                      memlet=dace.Memlet("Y[n, m]"))

            # state.add_memlet_path(vect_data,
            #                       write_map_entry,
            #                       tasklet,
            #                       dst_conn="from_kernel",
            #                       memlet=dace.Memlet("vec_data_C[m1]"))
            # pay attention if C has a single dimension (could be the case of batch =1)

        def make_compute(sdfg, state, vec_width=1):

            vec_type = dace.vector(dace.float32, vec_width)
            A_pipe_in = state.add_read("A_pipe")
            A_pipe_out = state.add_write("A_pipe")
            B_pipe_in = state.add_read("B_pipe")
            B_pipe_out = state.add_write("B_pipe")
            C_pipe_in = state.add_read("C_pipe")
            C_pipe_out = state.add_write("C_pipe")

            entry_n0, exit_n0 = state.add_map(
                "n0", {
                    "n0": "0:{}/{}".format(N, P),
                },
                schedule=dace.ScheduleType.FPGA_Device)
            entry_k, exit_k = state.add_map(
                "k", {"k": "0:{}".format(K)},
                schedule=dace.ScheduleType.FPGA_Device)
            entry_a, exit_a = state.add_map(
                "buffer_A", {"n1": "0:{}".format(P)},
                schedule=dace.ScheduleType.FPGA_Device)

            # As we are using vectorized data types for B, we have to consider it into these
            # two maps
            entry_m, exit_m = state.add_map(
                "m", {"m": "0:{}".format(M_Y, )},
                schedule=dace.ScheduleType.FPGA_Device)
            entry_c, exit_c = state.add_map(
                "write_C",
                {
                    "n1": "0:{}".format(P),
                    "m": "0:{}".format(M_Y)  # consider vectorization
                },
                schedule=dace.ScheduleType.FPGA_Device)

            # Instantiate buffers
            sdfg.add_scalar("A_reg",
                            dtype=dace.float32,
                            transient=True,
                            storage=dace.dtypes.StorageType.FPGA_Registers)
            A_reg = state.add_write("A_reg")

            # For C result we are going to use vectorized data type
            sdfg.add_array("C_buffer", [M_Y],
                           dtype=vec_type,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Local)
            C_buffer_in = state.add_read("C_buffer")
            C_buffer_out = state.add_write("C_buffer")

            # every PE: reads input data, buffer the data assigned to it, forwards the data
            buffer_a_tasklet = state.add_tasklet(
                "buffer_a", {"a_in"}, {"a_reg", "a_out"}, """\
if n1 == {P} - p - 1:
    a_reg = a_in
if p < {P} - 1:
    a_out = a_in""".format(P=P))
            state.add_memlet_path(A_pipe_in,
                                  entry_n0,
                                  entry_k,
                                  entry_a,
                                  buffer_a_tasklet,
                                  memlet=dace.Memlet("A_pipe[p]",
                                                     dynamic=False),
                                  dst_conn="a_in")
            state.add_memlet_path(buffer_a_tasklet,
                                  exit_a,
                                  A_reg,
                                  memlet=dace.Memlet("A_reg[0]", dynamic=True),
                                  src_conn="a_reg")
            state.add_memlet_path(buffer_a_tasklet,
                                  exit_a,
                                  exit_k,
                                  exit_n0,
                                  A_pipe_out,
                                  memlet=dace.Memlet("A_pipe[p + 1]",
                                                     dynamic=True),
                                  src_conn="a_out")
            # Compute and forward B
            compute_tasklet = state.add_tasklet(
                "multiply_add", {"a_in", "b_in", "c_in"}, {"b_out", "c_out"},
                """\
c_prev = 0 if k == 0 else c_in
c_out = c_prev + a_in * b_in
if p < {P} - 1:
    b_out = b_in""".format(P=P))

            state.add_memlet_path(A_reg,
                                  entry_m,
                                  compute_tasklet,
                                  dst_conn="a_in",
                                  memlet=dace.Memlet("A_reg[0]"))
            state.add_memlet_path(B_pipe_in,
                                  entry_n0,
                                  entry_k,
                                  entry_m,
                                  compute_tasklet,
                                  memlet=dace.Memlet("B_pipe[p]",
                                                     dynamic=False),
                                  dst_conn="b_in")
            state.add_memlet_path(compute_tasklet,
                                  exit_m,
                                  exit_k,
                                  exit_n0,
                                  B_pipe_out,
                                  memlet=dace.Memlet("B_pipe[p + 1]",
                                                     dynamic=True),
                                  src_conn="b_out")
            state.add_memlet_path(C_buffer_in,
                                  entry_k,
                                  entry_m,
                                  compute_tasklet,
                                  dst_conn="c_in",
                                  memlet=dace.Memlet("C_buffer[m]"))
            state.add_memlet_path(entry_n0, C_buffer_in, memlet=dace.Memlet())
            state.add_memlet_path(compute_tasklet,
                                  exit_m,
                                  exit_k,
                                  C_buffer_out,
                                  memlet=dace.Memlet("C_buffer[m]"),
                                  src_conn="c_out")
            state.add_memlet_path(C_buffer_out, exit_n0, memlet=dace.Memlet())

            write_c_tasklet = state.add_tasklet(
                "write_c", {"buffer_in", "forward_in"}, {"c_out"}, """\
if n1 <= p:
    c_out = forward_in if p > 0 and n1 > 0 else buffer_in""")
            state.add_memlet_path(C_buffer_out,
                                  entry_c,
                                  write_c_tasklet,
                                  memlet=dace.Memlet("C_buffer[m]",
                                                     dynamic=True),
                                  dst_conn="buffer_in")
            state.add_memlet_path(C_pipe_in,
                                  entry_n0,
                                  entry_c,
                                  write_c_tasklet,
                                  memlet=dace.Memlet("C_pipe[p-1]",
                                                     dynamic=True),
                                  dst_conn="forward_in")
            state.add_memlet_path(write_c_tasklet,
                                  exit_c,
                                  exit_n0,
                                  C_pipe_out,
                                  memlet=dace.Memlet("C_pipe[p]",
                                                     dynamic=True),
                                  src_conn="c_out")

            # Unroll processing elements
            compute_entry, compute_exit = state.add_map(
                "unroll_compute", {"p": "0:{}".format(P)},
                schedule=dace.ScheduleType.FPGA_Device,
                unroll=True)

            # Bring data nodes into scope
            state.add_memlet_path(compute_entry,
                                  A_pipe_in,
                                  memlet=dace.memlet.Memlet())
            state.add_memlet_path(compute_entry,
                                  B_pipe_in,
                                  memlet=dace.memlet.Memlet())
            state.add_memlet_path(compute_entry,
                                  C_pipe_in,
                                  memlet=dace.memlet.Memlet())
            state.add_memlet_path(A_pipe_out,
                                  compute_exit,
                                  memlet=dace.memlet.Memlet())
            state.add_memlet_path(B_pipe_out,
                                  compute_exit,
                                  memlet=dace.memlet.Memlet())
            state.add_memlet_path(C_pipe_out,
                                  compute_exit,
                                  memlet=dace.memlet.Memlet())

        # build the compute State
        vec_type = dace.vector(dace.float32, vec_width)

        new_sdfg.add_stream("A_pipe",
                            dace.float32,
                            transient=True,
                            shape=(P + 1, ),
                            storage=dace.dtypes.StorageType.FPGA_Local,
                            buffer_size=str(P))
        new_sdfg.add_stream("B_pipe",
                            vec_type,
                            transient=True,
                            shape=(P + 1, ),
                            storage=dace.dtypes.StorageType.FPGA_Local)
        new_sdfg.add_stream("C_pipe",
                            vec_type,
                            transient=True,
                            shape=(P + 1, ),
                            storage=dace.dtypes.StorageType.FPGA_Local)

        make_read_A(new_state)
        make_read_B(new_state, new_sdfg, vec_width)
        make_compute(new_sdfg, new_state, vec_width)
        make_write_C(new_state, new_sdfg, vec_width)

        new_sdfg.fill_scope_connectors()
        # Specialize the new sdfg, by using the input shapes
        new_sdfg.save("/tmp/gemm.sdfg")
        new_sdfg.validate()
        return new_sdfg


@autoregister_params(op="Reshape", name="fpga")
class PureReshape(ONNXForward):
    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        node.validate(sdfg, state)
        if (in_desc_with_name(node, state, sdfg, "data").dtype !=
                out_desc_with_name(node, state, sdfg, "reshaped")):
            raise ValueError(
                "Expected input and output to have the same dtype.")

        expansion = dace.SDFG("_reshape_expansion_")
        expansion.add_datadesc(
            "shape",
            copy.deepcopy(in_desc_with_name(node, state, sdfg, "shape")))
        indata = in_desc_with_name(node, state, sdfg, "data")
        outdata = out_desc_with_name(node, state, sdfg, "reshaped")
        expansion.add_datadesc("data", copy.deepcopy(indata))
        expansion.add_datadesc("reshaped", copy.deepcopy(outdata))
        expansion.arrays["shape"].transient = False
        expansion.arrays["data"].transient = False
        expansion.arrays["reshaped"].transient = False
        state = expansion.add_state()

        #TODO
        # ad hoc for lenet
        assert (len(indata.shape) == 4)
        assert (len(outdata.shape) == 2)
        map_ranges = {
            '__i%d' % i: '0:%s' % n
            for i, n in enumerate(indata.shape)
        }
        me, mx = state.add_map("reshaping", map_ranges)
        tasklet = state.add_tasklet('reshape_task', ['_in'], ['_out'],
                                    '_out = _in')

        data = state.add_read("data")
        reshaped = state.add_write("reshaped")
        state.add_memlet_path(data,
                              me,
                              tasklet,
                              dst_conn="_in",
                              memlet=dace.Memlet("data[{}]".format(",".join([
                                  '__i%d' % i for i in range(len(indata.shape))
                              ]))))
        state.add_memlet_path(
            tasklet,
            mx,
            reshaped,
            src_conn="_out",
            memlet=dace.Memlet(
                "reshaped[__i0, __i1*{} + __i2*{} +__i3 ]".format(
                    indata.shape[2] * indata.shape[3], indata.shape[3])))
        # memlet = expansion.make_array_memlet("data")
        # memlet.allow_oob = True

        # state.add_edge(data, None, reshaped, None, memlet)
        expansion.fill_scope_connectors()
        return expansion


@autoregister_params(op="Softmax", name="fpga")
class PureSoftmax(ONNXForward):
    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:
        # FIRST ATTEMPT
        # try to avoid max computation, this could have
        # problems for numerical stability
        # https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
        # result = exp / sum

        node.validate(sdfg, state)
        inparr = in_desc_with_name(node, state, sdfg, "input")
        outarr = out_desc_with_name(node, state, sdfg, "output")

        axis = node.axis
        if type(axis) is not int or not (-len(inparr.shape) <= axis < len(
                inparr.shape)):
            raise ValueError("expected axis to be an integer in range"
                             " [-{}, {}), got {}".format(
                                 len(inparr.shape), len(inparr.shape), axis))

        if axis < 0:
            axis += len(inparr.shape)
        out_tmp_shape = inparr.shape
        out_tmp_dtype = inparr.dtype

        #ad hoc lenet implementation, needs to be generalized
        assert (len(inparr.shape) == 2)

        new_sdfg = dace.SDFG("fpga_softmax")
        new_state = new_sdfg.add_state("compute")
        new_sdfg.add_datadesc("input", copy.deepcopy(inparr))
        new_sdfg.add_datadesc("output", copy.deepcopy(outarr))

        # Add registers to store exp results
        # NOTE: ok in lenet since we are not working with large input size
        new_sdfg.add_array("exp_data", [inparr.shape[-1]],
                           dtype=dace.float32,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Registers)
        new_sdfg.add_array("sum_data", [1],
                           dtype=dace.float32,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Registers)

        ##################
        # exp of all elements, store them into registers

        # Create a two level maps: outermost is for each batch element
        # Inside we will have two maps, one after the other, that computes
        # the exp and the div

        #batch map
        batch_me, batch_mx = new_state.add_map(
            "softmax_batch", dict(b="0:{}".format(inparr.shape[0])))

        #exp map
        exp_me, exp_mx = new_state.add_map(
            "softmax_exp", dict(i="0:{}".format(inparr.shape[-1])))

        #div map
        div_me, div_mx = new_state.add_map(
            "softmax_max", dict(i="0:{}".format(inparr.shape[-1])))

        exp_tasklet = new_state.add_tasklet(
            'exp_task',
            ['_in', '_in_sum'],
            ['_out', '_out_sum'],
            '_exp = float(0)\n'  #for type inference
            '_exp = exp(_in)\n'
            'prev_sum = _in_sum if i!=0 else float(0)\n'
            '_out_sum = prev_sum + _exp\n'
            '_out = _exp')
        div_tasklet = new_state.add_tasklet('div_task', ['_in', '_sum'],
                                            ['_out'], '_out = _in/_sum')

        in_read = new_state.add_read("input")
        out_write = new_state.add_write("output")
        exp_data = new_state.add_access("exp_data")
        sum_in = new_state.add_access("sum_data")
        sum_accum = new_state.add_access("sum_data")
        init_tasklet = new_state.add_tasklet('init_task', [], ['_out'],
                                             '_out = float(0)')

        new_state.add_memlet_path(in_read,
                                  batch_me,
                                  exp_me,
                                  exp_tasklet,
                                  dst_conn="_in",
                                  memlet=dace.Memlet("input[b,i]"))

        new_state.add_memlet_path(init_tasklet,
                                  sum_in,
                                  src_conn="_out",
                                  memlet=dace.Memlet("sum_data[0]"))

        new_state.add_memlet_path(sum_in,
                                  exp_me,
                                  exp_tasklet,
                                  dst_conn="_in_sum",
                                  memlet=dace.Memlet("sum_data[0]"))
        new_state.add_memlet_path(batch_me, init_tasklet, memlet=dace.Memlet())
        new_state.add_memlet_path(exp_tasklet,
                                  exp_mx,
                                  exp_data,
                                  src_conn="_out",
                                  memlet=dace.Memlet("exp_data[i]"))
        new_state.add_memlet_path(exp_tasklet,
                                  exp_mx,
                                  sum_accum,
                                  src_conn="_out_sum",
                                  memlet=dace.Memlet("sum_data[0]"))

        ###### DIV

        new_state.add_memlet_path(exp_data,
                                  div_me,
                                  div_tasklet,
                                  dst_conn="_in",
                                  memlet=dace.Memlet("exp_data[i]"))

        new_state.add_memlet_path(sum_accum,
                                  div_me,
                                  div_tasklet,
                                  dst_conn="_sum",
                                  memlet=dace.Memlet("sum_data[0]"))
        new_state.add_memlet_path(div_tasklet,
                                  div_mx,
                                  batch_mx,
                                  out_write,
                                  src_conn="_out",
                                  memlet=dace.Memlet("output[b, i]"),
                                  propagate=False)

        new_sdfg.fill_scope_connectors()
        new_sdfg.save('/tmp/softmax.sdfg')
        return new_sdfg