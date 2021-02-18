'''
This file contains generic fpga node expansions.
Here generic refer to the fact that input/output shape can be symbols,
so we can not assume anything on them

'''

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


def derive_shape_and_strides_for_symbolic_data(data, symbolic_axis: list):
    '''
    Returns the shape and strides of a data node with symbolic sizes

    NOTE: for the strides it assumes that the last one is the contiguous dimension
    :param data: node
           symbol_names: list of symbols that correspond to the data axis
           ATTENTION: they have to be ordered
    :return:
    '''

    data_shape = []
    data_strides = []

    for i, n in enumerate(data.shape):
        # keep track of shapes and strides
        # the stride looks like [axis1*axis2, axis2, 1]
        data_shape.append(symbolic_axis[i])
        data_strides.append(1)
        for j in range(i):
            data_strides[j] = data_strides[j] * symbolic_axis[i]
    return tuple(data_shape), tuple(data_strides)


@autoregister_params(op="Relu", name="generic_fpga")
class FPGARelu(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        X = in_desc_with_name(node, state, sdfg, "X")
        Y = out_desc_with_name(node, state, sdfg, "Y")

        # # TODO: For the moment being, we support the same vect width
        # if X.veclen != Y.veclen:
        #     return False
        return True

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        X = in_desc_with_name(node, state, sdfg, "X")
        Y = out_desc_with_name(node, state, sdfg, "Y")

        vec_width = X.veclen

        streaming_node = False
        if X.veclen != Y.veclen:
            # we will need to copy the data out accordingly
            # NOTE: for the moment, tested with Y veclen = 1
            vec_width_mismatch = True
        else:
            vec_width_mismatch = False

        # Build map ranges: one loop per dimension. We use internal symbols here, that will
        # be mapped to external ones

        new_sdfg = dace.SDFG("fpga_relu")
        # Build symbolic map ranges, shape and strides: one loop per dimension
        # We use internal symbols
        map_ranges = dict()
        x_shape = []
        y_shape = []
        x_strides = []
        y_strides = []
        x_axis_names = []
        y_axis_names = []

        for i, n in enumerate(X.shape):
            axis_name = "relu_x_axis{}".format(i)
            x_axis_names.append(axis_name)
            map_ranges[f"__i{i}"] = f"0:{axis_name}"
            # add the symbol
            symbolic_axis = dace.symbol(axis_name)
            new_sdfg.add_symbol(symbolic_axis.name, dace.int32)
            # keep track of shapes and strides
            # note that the stride looks like [relu_axis1*relu_axis2, relu_axis2, 1]
            x_shape.append(symbolic_axis)
            x_strides.append(1)
            for j in range(i):
                x_strides[j] = x_strides[j] * symbolic_axis

        for i, n in enumerate(Y.shape):
            axis_name = "relu_y_axis{}".format(i)
            y_axis_names.append(axis_name)
            # add the symbol
            symbolic_axis = dace.symbol(axis_name)
            new_sdfg.add_symbol(symbolic_axis.name, dace.int32)
            # keep track of shapes and strides
            # note that the stride looks like [relu_axis1*relu_axis2, relu_axis2, 1]
            y_shape.append(symbolic_axis)
            y_strides.append(1)
            for j in range(i):
                y_strides[j] = y_strides[j] * symbolic_axis

        new_state = new_sdfg.add_state("compute")
        # Create local versions of input data nodes, but using our new symbols, not the real ones
        # in order to not use external shapes. Later we will do symbol mapping
        new_sdfg.add_array("X",
                           shape=x_shape,
                           dtype=X.dtype,
                           storage=X.storage,
                           strides=x_strides,
                           transient=False)

        new_sdfg.add_array("Y",
                           shape=y_shape,
                           dtype=Y.dtype,
                           storage=Y.storage,
                           strides=y_strides,
                           transient=False)

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
                'relu_write_out_map', dict(i="0:{}".format(vec_width)), unroll=True)
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
                memlet=dace.Memlet("Y[__i0, __i1, __i2, __i3*{}+i]".format(vec_width)))

        else:
            #write out
            new_state.add_memlet_path(
                vec_data_out,
                outer_mx,
                y_write,
                memlet=dace.Memlet("Y[{}]".format(",".join(
                    ['__i%d' % i for i in range(len(X.shape))]))))
        new_sdfg.fill_scope_connectors()

        # Modify internal schedules according to node schedule
        # TODO: is this needed?
        if node.schedule != dace.ScheduleType.Default:
            for nstate in new_sdfg.nodes():
                for topnode, scope in nstate.scope_dict().items():
                    if scope is None and isinstance(
                            topnode,
                        (dace.nodes.EntryNode, dace.nodes.LibraryNode)):
                        topnode.schedule = node.schedule

        # symbols = sdfg.free_symbols
        # symbol_mapping = {s: s for s in symbols}
        symbol_mapping = dict()
        new_sdfg.save("/tmp/newsdfg.sdfg")

        # nest and map symbol
        for i, n in enumerate(X.shape):
            symbol_mapping[x_axis_names[i]] = n

        for i, n in enumerate(Y.shape):
            symbol_mapping[y_axis_names[i]] = n

        expansion = state.add_nested_sdfg(new_sdfg,
                                          sdfg,
                                          node.in_connectors,
                                          node.out_connectors,
                                          name=node.name,
                                          debuginfo=node.debuginfo,
                                          symbol_mapping=symbol_mapping)

        expansion.sdfg.save('/tmp/relu_expansion.sdfg')
        return expansion


@autoregister_params(op="MaxPool", name="generic_fpga")
class FPGAMaxPool2D(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        X = in_desc_with_name(node, state, sdfg, "X")
        Y = out_desc_with_name(node, state, sdfg, "Y")

        if Y.veclen != 1:  #NYI
            return False

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

        # This implementations do not use shift registers
        # It will have nested maps, with the innermost referring to the
        # filter size and unrolled (for the moment being)

        # TODO: remove unrolled reads/write from/to memory

        X = in_desc_with_name(node, state, sdfg, "X")
        Y = out_desc_with_name(node, state, sdfg, "Y")
        vec_width = X.veclen

        image_dims = len(X.shape) - 2
        # batch_size = X.shape[0]
        # num_channels = X.shape[1]
        strides = node.strides if node.strides is not None else [
            1 for _ in range(image_dims)
        ]
        #TODO: symbolize
        stride_height, stride_width = strides
        filter_height, filter_width = node.kernel_shape
        # input_size_height, input_size_width = X.shape[2:]
        # output_size_y, output_size_x = Y.shape[2:]

        new_sdfg = dace.SDFG("fpga_maxpool")
        new_state = new_sdfg.add_state("compute")

        batch_size = dace.symbol("batch_size")
        input_channels = dace.symbol("input_channels")
        input_height = dace.symbol("input_height")
        input_width = dace.symbol("input_width")
        output_height = dace.symbol("output_height")
        output_width = dace.symbol("output_width")
        new_sdfg.add_symbol(batch_size.name, dace.int32)
        new_sdfg.add_symbol(input_channels.name, dace.int32)
        new_sdfg.add_symbol(input_height.name, dace.int32)
        new_sdfg.add_symbol(input_width.name, dace.int32)
        new_sdfg.add_symbol(output_height.name, dace.int32)
        new_sdfg.add_symbol(output_width.name, dace.int32)

        new_sdfg.add_array(
            "X",
            shape=(batch_size, input_channels, input_height, input_width),
            dtype=X.dtype,
            storage=X.storage,
            strides=(input_channels * input_height * input_width,
                     input_height * input_width, input_width, 1),
            transient=False)

        new_sdfg.add_array(
            "Y",
            shape=(batch_size, input_channels, output_height, output_width),
            dtype=Y.dtype,
            storage=Y.storage,
            strides=(input_channels * output_height * output_width,
                     output_height * output_width, output_width, 1),
            transient=False)

        # new_sdfg.add_datadesc("X", copy.deepcopy(X))
        # new_sdfg.add_datadesc("Y", copy.deepcopy(Y))
        #
        # new_sdfg.arrays["X"].transient = False
        # new_sdfg.arrays["Y"].transient = False

        # variable for reduction
        new_sdfg.add_array("max_res", [1],
                           dace.float32,
                           storage=dace.StorageType.FPGA_Registers,
                           transient=True)
        new_sdfg.add_array('vec_data',
                           shape=[
                               vec_width,
                           ],
                           dtype=dace.float32,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Registers)
        # temporary storage for unpacked vector data type

        # the outer map loops over every entry of the input array

        outer_me, outer_mx = new_state.add_map(
            'outer_pool_map',
            dict(b="0:{}".format(batch_size),
                 c="0:{}".format(input_channels),
                 out_y="0:{}".format(output_height),
                 out_x="0:{}".format(output_width)))

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
            code="if hx == 0 and hy == 0: max_in = {}\n"  #init
            "max_out = float(max(max_in, image_in))\n"
            "if hy == {} - 1 and hx == {} -1: output = max_out".format(
                dtypes.min_value(Y.dtype), filter_height, filter_width,
                filter_height, filter_height, vec_width, filter_height,
                filter_width))

        read_X = new_state.add_read("X")
        write_Y = new_state.add_write("Y")
        read_max_res = new_state.add_access("max_res")
        write_max_res = new_state.add_access("max_res")

        # memlets: input data

        new_state.add_memlet_path(
            read_X,
            outer_me,
            inner_me,
            compute_tasklet,
            dst_conn="image_in",
            memlet=dace.Memlet("X[b, c, out_y*{}+hy, out_x*{}+hx]".format(
                filter_height, filter_height)))
        #memlets for max
        new_state.add_memlet_path(read_max_res,
                                  inner_me,
                                  compute_tasklet,
                                  dst_conn="max_in",
                                  memlet=dace.Memlet("max_res[0]"))

        new_state.add_memlet_path(compute_tasklet,
                                  inner_mx,
                                  write_max_res,
                                  src_conn="max_out",
                                  memlet=dace.Memlet("max_res[0]"))

        # Attention: use propagate=False otherwise it does not validate
        new_state.add_memlet_path(compute_tasklet,
                                  inner_mx,
                                  outer_mx,
                                  write_Y,
                                  src_conn="output",
                                  memlet=dace.Memlet("Y[b,c,out_y, out_x]"),
                                  propagate=False)
        #put max_res in scope
        new_state.add_memlet_path(outer_me, read_max_res, memlet=dace.Memlet())
        new_state.add_memlet_path(write_max_res,
                                  outer_mx,
                                  memlet=dace.Memlet())

        new_sdfg.fill_scope_connectors()

        # nest the sdfg and map symbols
        symbol_mapping = {
            "batch_size": X.shape[0],
            "input_channels": X.shape[1],
            "input_height": X.shape[2],
            "input_width": X.shape[3],
            "output_height": Y.shape[2],
            "output_width": Y.shape[3]
        }

        expansion = state.add_nested_sdfg(new_sdfg,
                                          sdfg,
                                          node.in_connectors,
                                          node.out_connectors,
                                          name=node.name,
                                          debuginfo=node.debuginfo,
                                          symbol_mapping=symbol_mapping)

        expansion.sdfg.save('/tmp/expansion.sdfg')
        return expansion


@autoregister_params(op="Gemm", name="generic_fpga")
class FPGAGemm(ONNXForward):
    '''
        GEMM implementation. For the moment being it covers only the case with A non transposed and B transposed
        The base implementation resemble the Systolic GEMM FPGA DaCe sample

        This is a generic implementation:
        - it has fixed # of PEs and tile sizes
        - it properly deal with input that are not perfect multiple of them. If this is the case
            it will compute random data that will be not written in memory
        - it assumes that tile_size % vec_width == 0 (if that is not the case we should also read from memory not just write)
        - it assumes that M % vec_width ==0


        ATTENTION:
        - if used with input_to_constant it currently has some problem with dynamic memlet (they are overwritten in
            Intel FPGA Keyword Remover)
    '''
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

        batch_size = dace.symbol("batch_size")
        input_features = dace.symbol("input_features")
        output_features = dace.symbol("output_features")
        new_sdfg.add_symbol(batch_size.name, dace.int32)
        new_sdfg.add_symbol(input_features.name, dace.int32)
        new_sdfg.add_symbol(output_features.name, dace.int32)

        # Create local versions of input data nodes, but using our new symbols, not the real ones
        # in order to not use external shapes and complicated formulas that could pop out from shape inference.
        # Later we will do symbol mapping

        new_sdfg.add_array("A",
                           shape=(batch_size, input_features),
                           dtype=A.dtype,
                           storage=A.storage,
                           strides=(input_features, 1),
                           transient=False)

        new_sdfg.add_array("B",
                           shape=(output_features, input_features),
                           dtype=B.dtype,
                           storage=B.storage,
                           strides=(input_features, 1),
                           transient=False)

        new_sdfg.add_array("C",
                           shape=[output_features],
                           dtype=C.dtype,
                           storage=C.storage,
                           strides=[1],
                           transient=False)

        new_sdfg.add_array("Y",
                           shape=(
                               batch_size,
                               output_features,
                           ),
                           dtype=Y.dtype,
                           storage=Y.storage,
                           strides=(output_features, 1),
                           transient=False)

        # GEMM Parameters
        N = batch_size
        K = input_features
        M = output_features

        # The input can be vectorized along M. Update M to reflect this
        vec_width = Y.veclen
        M = M * vec_width

        # while the output Y can be vectorized
        # M_C = C.shape[0]
        # M_Y = Y.shape[1]

        # GEMM Computational Units parameters
        # TODO: these as libnode parameter
        # TODO: deal with dimensions that are not multiple of this

        P = 16  # Number of Processing Elements
        T = 128   # Tile size, not considering vectorization width (so plain floats)

        assert (T % vec_width == 0)

        #safe delay: apparently, we should wait at least 11 cycles to add on the same floating number
        # stored in BRAM (6 clock cycle for the madd, 4 clock cycles for accessing BRAM)
        L = max(11 - T//vec_width, 0)

        ####################################################
        # Build the SDFG: starting point: gemm_fpga_systolic vectorized sample

        def make_read_A(state):

            # TODO: vectorize also this, by reading more than one element at a time

            #Deal with M not a multiple of T
            entry, exit = state.add_map(
                "read_A",
                {
                    "n0": "0:ceiling({}/{})".format(N, P),
                    "tm": "0:ceiling({}/{})".format(
                        M, T),  # must be repeated according to the tile size
                    "k": "0:{}".format(K)
                },
                schedule=dace.ScheduleType.FPGA_Device)
            # use a different map, and unroll it if necessary
            unroll_inner_map = P > (T//vec_width + L) and P <= 16
            send_map_entry, send_map_exit = state.add_map(
                "send_A", {"n1": "0:{}".format(P)},
                schedule=dace.ScheduleType.FPGA_Device,
                unroll=unroll_inner_map)

            mem = state.add_read("A")
            pipe = state.add_write("A_pipe")
            tasklet = state.add_tasklet("read_A", {"from_memory"},
                                        {"to_kernel"},"""\
data = from_memory if n0*{} + n1 < {} else 0
to_kernel = from_memory""".format(P,N))

            state.add_memlet_path(mem,
                                  entry,
                                  send_map_entry,
                                  tasklet,
                                  dst_conn="from_memory",
                                  memlet=dace.Memlet(
                                      "A[n0 * {} + n1, k]".format(P), dynamic=True, allow_oob=True))
            state.add_memlet_path(tasklet,
                                  send_map_exit,
                                  exit,
                                  pipe,
                                  src_conn="to_kernel",
                                  memlet=dace.Memlet(
                                      "A_pipe[{} - n1 - 1]".format(P)))

        def make_read_B(state, sdfg, vec_width=1):

            # NOTE: We are reading this transposed: B is originally a matrix MxK

            # B is accessed by row for the GEMM in LENET
            # gear boxing: we read plain data types, we stream vector data types
            # Therefore we have two maps, the innermost is unrolled
            entry, exit = state.add_map(
                "read_B",
                {
                    "n": "0:ceiling({}/{})".format(N, P),
                    "tm": "0:ceiling({}/{})".format(
                        M, T),  # M already consider vec_width
                    "k": "0:{}".format(K),
                    "m0": "0:{}/{}".format(T, vec_width)
                },
                schedule=dace.ScheduleType.FPGA_Device)

            read_map_entry, read_map_exit = state.add_map(
                "unrolled_reads_B", {"m1": "0:{}".format(vec_width)},
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
            tasklet = state.add_tasklet(
                "read_B", {"from_memory"}, {"to_kernel"}, """\
data = from_memory if tm*{} + m0*{} + m1 < {} else float(0)
to_kernel = data """.format(T, vec_width, M))

            # In the innermost map we read W=vec_width data elements and we store them into `vec_data`
            state.add_memlet_path(mem,
                                  entry,
                                  read_map_entry,
                                  tasklet,
                                  memlet=dace.Memlet(
                                      "B[tm*{} +m0*{}+m1, k]".format(
                                          T, vec_width),
                                      dynamic=True),
                                  dst_conn="from_memory")

            state.add_memlet_path(tasklet,
                                  read_map_exit,
                                  vect_data,
                                  src_conn="to_kernel",
                                  memlet=dace.Memlet("vec_data_B[m1]",
                                                     dynamic=True))

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
                    "n0": "0:ceiling({}/{})".format(N, P),
                    "tm": "0:ceiling({}/{})".format(M, T),
                    "n1": "0:{}".format(P),
                    "m": "0:{}/{}".format(
                        T, vec_width)  #consider also vectorization
                },
                schedule=dace.ScheduleType.FPGA_Device)

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
                add_C_tasklet = state.add_tasklet(
                    'add_C_tasklet', {'in_con', 'prev_c'}, {'out_con'}, """\
if tm * {} + m * {} + m1 < {}  and  n0*{}+n1 < {} :                                               
    out_con = in_con + prev_c """.format(T, vec_width, M, P, N))
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
                                          "C[tm*{} + m*{} + m1]".format(
                                              T, vec_width),
                                          dynamic=True))

                state.add_memlet_path(add_C_tasklet,
                                      add_map_exit,
                                      vect_res,
                                      src_conn="out_con",
                                      memlet=dace.Memlet("vec_res[m1]"))
                # write out: we have to write out only if we are not after M
                # here we exploit the fact that M%vec_width == 0 (By construction, see above)
                # we need an additional tasklet, so that we can have the dynamic memlet
                write_tasklet = state.add_tasklet(
                    'write', {'in_con'}, {'out_con'}, """\
if tm * {} + m * {}  < {}  and  n0*{}+n1 < {}:
    out_con = in_con""".format(T, vec_width, M, P, N))
                state.add_memlet_path(vect_res,
                                      write_tasklet,
                                      dst_conn="in_con",
                                      memlet=dace.Memlet(
                                          "vec_res[0:{}]".format(vec_width)))

                state.add_memlet_path(write_tasklet,
                                      exit_map,
                                      mem,
                                      src_conn="out_con",
                                      memlet=dace.Memlet(
                                          "Y[n0*{}+n1, tm*{}/{} + m]".format(
                                              P, T, vec_width),
                                          dynamic=True))

            else:
                tasklet = state.add_tasklet(
                    "write_C", {"from_kernel", "prev_c"}, {"to_memory"}, """\
if tm * {} + m < {}  and  n0*{}+n1 < {}:                 
    to_memory = (from_kernel + prev_c)""".format(T, M, P, N))
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
                                      memlet=dace.Memlet(
                                          "C[tm*{} + m]".format(T),
                                          dynamic=True))
                state.add_memlet_path(
                    tasklet,
                    exit_map,
                    mem,
                    src_conn="to_memory",
                    memlet=dace.Memlet("Y[n0*{}+n1, tm*{} + m]".format(P, T),
                                       dynamic=True))

        def make_compute(sdfg, state, vec_width=1):

            vec_type = dace.vector(dace.float32, vec_width)
            A_pipe_in = state.add_read("A_pipe")
            # A_pipe_out = state.add_write("A_pipe")
            B_pipe_in = state.add_read("B_pipe")
            B_pipe_out = state.add_write("B_pipe")
            C_pipe_in = state.add_read("C_pipe")
            C_pipe_out = state.add_write("C_pipe")

            entry_pipeline, exit_pipeline = state.add_pipeline(
                "compute_and_drain",
                {
                    "n0": "0:ceiling({}/{})".format(N, P),
                    "tm": "0:ceiling({}/{})".format(
                        M, T),  #if this is vectorized, M accounts for that
                    "k": "0:{}".format(K),
                    "m": "0:{}/{} + {}".format(T, vec_width, L)
                },
                drain_size=P * T // vec_width,
                drain_overlap=False,
                additional_iterators={
                    'm_drain': 0,
                    'k_drain': 0
                },
                schedule=dace.ScheduleType.FPGA_Device)

            # Instantiate buffers
            sdfg.add_scalar("A_reg",
                            dtype=dace.float32,
                            transient=True,
                            storage=dace.dtypes.StorageType.FPGA_Registers)
            A_reg = state.add_write("A_reg")
            A_reg_init = state.add_access("A_reg")

            # For C result we are going to use vectorized data type

            # Note: for some of the Sacred Mysteries of Intel OpenCL Compiler (TM), if this buffer is smaller
            # than 24 floats, the II of the pipeline will be 5. Therefore we check this (with 32 to be
            # more compliant with standard vector size) and in case we enlarge it

            buffer_size = max(T, 32) / vec_width
            sdfg.add_array("C_buffer", [buffer_size],
                           dtype=vec_type,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Local)
            C_buffer_in = state.add_read("C_buffer")
            C_buffer_out = state.add_write("C_buffer")

            # Feed A
            # every PE: reads input data, buffer the data assigned to it
            buffer_a_tasklet = state.add_tasklet(
                "buffer_a", {"a_in"}, {
                    "a_reg",
                }, """\
if m == 0 and not {}:
    a_reg = a_in""".format(entry_pipeline.pipeline.drain_condition()))
            state.add_memlet_path(A_pipe_in,
                                  entry_pipeline,
                                  buffer_a_tasklet,
                                  memlet=dace.Memlet("A_pipe[p]",
                                                     dynamic=True),
                                  dst_conn="a_in")
            state.add_memlet_path(buffer_a_tasklet,
                                  A_reg,
                                  memlet=dace.Memlet("A_reg[0]", dynamic=True),
                                  src_conn="a_reg")

            # Feed B
            # Read B: done outside of the compute tasklet to help type inference
            sdfg.add_array("B_reg",
                           shape=[1],
                           dtype=vec_type,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Local)
            B_reg = state.add_access("B_reg")
            buffer_b_tasklet = state.add_tasklet(
                "buffer_b", {"b_in"}, {"b_reg_out"}, """\
if  m>={} and not {}:
    b_reg_out = b_in""".format(L, entry_pipeline.pipeline.drain_condition()))

            state.add_memlet_path(B_pipe_in,
                                  entry_pipeline,
                                  buffer_b_tasklet,
                                  memlet=dace.Memlet("B_pipe[p]",
                                                     dynamic=True),
                                  dst_conn="b_in")
            state.add_memlet_path(buffer_b_tasklet,
                                  B_reg,
                                  memlet=dace.Memlet("B_reg[0]", dynamic=True),
                                  src_conn="b_reg_out")
            # COMPUTE AND DRAIN
            # Compute and forward B: this is done if we are not in the init phase of the pipeline
            compute_tasklet = state.add_tasklet(
                "compute_and_drain", {"a_in", "b_in", "c_in", "forward_in"},
                {"b_out", "c_out", "c_pipe_out"}, f"""\
if m>= {L} and not {entry_pipeline.pipeline.drain_condition()}:
    c_prev = 0 if k == 0 else c_in     
    c_out =  c_prev + a_in * b_in
    if p < {P} - 1:
        b_out = b_in
# Drain
# when we have to drain:
# - if k = K-1 and m>=L: drain my own result
#-  otherwise, if k_drain<p forward data coming from previous PEs (this could happens also in the drain phase)
if((n0 > 0 or tm > 0)  and k_drain <p and m_drain <{T}/{vec_width}) or  (k=={K}-1 and m>= {L}) or ({entry_pipeline.pipeline.drain_condition()} and k_drain < p):
   c_pipe_out = c_out if (p==0 or (k_drain=={K}-1 and not {entry_pipeline.pipeline.drain_condition()})) else forward_in

# adjust draining iterators
if not {entry_pipeline.pipeline.drain_condition()}:
    if m_drain >= {L} +  {T}/{vec_width} -1:
        m_drain = 0
        if k_drain >= {K} - 1:
            k_drain = 0
        else:
            k_drain = k_drain +1
    else:
        m_drain = m_drain + 1
else:
    if m_drain >=  {T}/{vec_width} -1:
        m_drain = 0
        if k_drain >= {K} - 1:
            k_drain = 0
        else:
            k_drain = k_drain +1
    else:
        m_drain = m_drain + 1
            """)

            state.add_memlet_path(A_reg,
                                  compute_tasklet,
                                  dst_conn="a_in",
                                  memlet=dace.Memlet("A_reg[0]"))
            state.add_memlet_path(B_reg,
                                  compute_tasklet,
                                  memlet=dace.Memlet("B_reg[0]",
                                                     dynamic=False),
                                  dst_conn="b_in")

            state.add_memlet_path(compute_tasklet,
                                  exit_pipeline,
                                  B_pipe_out,
                                  memlet=dace.Memlet("B_pipe[p + 1]",
                                                     dynamic=True),
                                  src_conn="b_out")
            state.add_memlet_path(C_buffer_in,
                                  entry_pipeline,
                                  compute_tasklet,
                                  dst_conn="c_in",
                                  memlet=dace.Memlet(
                                      "C_buffer[m-{}]".format(L),
                                      allow_oob=True))

            state.add_memlet_path(compute_tasklet,
                                  exit_pipeline,
                                  C_buffer_out,
                                  memlet=dace.Memlet(
                                      "C_buffer[m-{}]".format(L),
                                      allow_oob=True,
                                      dynamic=True),
                                  src_conn="c_out")

            state.add_memlet_path(C_pipe_in,
                                  entry_pipeline,
                                  compute_tasklet,
                                  memlet=dace.Memlet("C_pipe[p-1]",
                                                     dynamic=True),
                                  dst_conn="forward_in")
            state.add_memlet_path(compute_tasklet,
                                  exit_pipeline,
                                  C_pipe_out,
                                  memlet=dace.Memlet("C_pipe[p]",
                                                     dynamic=True),
                                  src_conn="c_pipe_out")

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
            # state.add_memlet_path(A_pipe_out,
            #                       compute_exit,
            #                       memlet=dace.memlet.Memlet())
            state.add_memlet_path(B_pipe_out,
                                  compute_exit,
                                  memlet=dace.memlet.Memlet())
            state.add_memlet_path(C_pipe_out,
                                  compute_exit,
                                  memlet=dace.memlet.Memlet())

            state.add_memlet_path(compute_entry,
                                  A_reg_init,
                                  memlet=dace.memlet.Memlet())
            state.add_memlet_path(A_reg_init,
                                  entry_pipeline,
                                  memlet=dace.memlet.Memlet())
            b_init = state.add_access("B_reg")
            state.add_memlet_path(compute_entry, b_init, memlet=dace.Memlet())
            state.add_memlet_path(b_init, entry_pipeline, memlet=dace.Memlet())
            state.add_memlet_path(compute_entry,
                                  C_buffer_in,
                                  memlet=dace.Memlet())

        # build the compute State
        vec_type = dace.vector(dace.float32, vec_width)

        new_sdfg.add_stream("A_pipe",
                            dace.float32,
                            transient=True,
                            shape=(P, ),
                            storage=dace.dtypes.StorageType.FPGA_Local,
                            buffer_size=str(P))
        new_sdfg.add_stream("B_pipe",
                            vec_type,
                            transient=True,
                            shape=(P + 1, ),
                            buffer_size=2,
                            storage=dace.dtypes.StorageType.FPGA_Local)
        new_sdfg.add_stream("C_pipe",
                            vec_type,
                            transient=True,
                            shape=(P + 1, ),
                            buffer_size=T,
                            storage=dace.dtypes.StorageType.FPGA_Local)

        make_read_A(new_state)
        make_read_B(new_state, new_sdfg, vec_width)
        make_compute(new_sdfg, new_state, vec_width)
        make_write_C(new_state, new_sdfg, vec_width)

        new_sdfg.fill_scope_connectors()
        # Specialize the new sdfg, by using the input shapes
        new_sdfg.save("/tmp/gemm.sdfg")
        new_sdfg.validate()

        # nest the sdfg and map symbols
        symbol_mapping = {
            "batch_size": A.shape[0],
            "input_features": A.shape[1],
            "output_features": Y.shape[1],
        }

        expansion = state.add_nested_sdfg(new_sdfg,
                                          sdfg,
                                          node.in_connectors,
                                          node.out_connectors,
                                          name=node.name,
                                          debuginfo=node.debuginfo,
                                          symbol_mapping=symbol_mapping)

        expansion.sdfg.save('/tmp/expansion.sdfg')
        return expansion


@autoregister_params(op="Conv", name="generic_fpga")
class FPGAGenericIm2ColConv(ONNXForward):
    """
    Conv implementation based on Gemm

    This is a generic implementation:
    - it has fixed # of PEs and tile sizes
    - it properly deal with input that are not perfect multiple of them. If this is the case
        it will compute random data that will be not written in memory
    - it assumes that tile_size % vec_width == 0 (if that is not the case we should also read from memory not just write)
    - it properly deals with the case in which some PE does nothing


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

        if node.kernel_shape is not None:
            filter_hy, filter_hx = node.kernel_shape
        else:
            filter_hy, filter_hx = W.shape[2:]

        new_sdfg = dace.SDFG("fpga_im2col_conv")

        # TODO:
        # -  can we use W.shape to detect also filter sizes?

        # Create local symbols and mapping with external one
        batch_size = dace.symbol("batch_size")
        num_channels = dace.symbol("num_channels")
        num_filters = dace.symbol("num_filters")
        #input size symbols are not really needed for the actual computation but we need them for the input shape
        input_size_x = dace.symbol("input_size_x")
        input_size_y = dace.symbol("input_size_y")
        output_size_x = dace.symbol("output_size_x")
        output_size_y = dace.symbol("output_size_y")
        output_size_y = dace.symbol("output_size_y")
        output_size_y = dace.symbol("output_size_y")
        new_sdfg.add_symbol(batch_size.name, dace.int32)
        new_sdfg.add_symbol(num_channels.name, dace.int32)
        new_sdfg.add_symbol(num_filters.name, dace.int32)
        new_sdfg.add_symbol(output_size_x.name, dace.int32)
        new_sdfg.add_symbol(output_size_y.name, dace.int32)
        new_sdfg.add_symbol(input_size_x.name, dace.int32)
        new_sdfg.add_symbol(input_size_y.name, dace.int32)

        symbol_mapping = {
            batch_size.name: X.shape[0],
            num_channels.name: X.shape[1],
            input_size_y.name: X.shape[2],
            input_size_x.name: X.shape[3],
            num_filters.name: W.shape[0],
            output_size_y.name: Y.shape[-2],
            output_size_x.name: Y.shape[-1]
        }

        X_shape, X_strides = derive_shape_and_strides_for_symbolic_data(
            X, [batch_size, num_channels, input_size_y, input_size_x])
        W_shape, W_strides = derive_shape_and_strides_for_symbolic_data(
            W, [num_filters, num_channels, filter_hy, filter_hx])
        B_shape = tuple([num_filters])
        B_strides = tuple([1])
        Y_shape, Y_strides = derive_shape_and_strides_for_symbolic_data(
            Y, [batch_size, num_filters, output_size_y, output_size_x])
        new_state = new_sdfg.add_state()

        # Create local versions of input data nodes, but using our new symbols, not the real ones
        # in order to not use external shapes and complicated formulas that could pop out from shape inference.
        # Later we will do symbol mapping
        new_sdfg.add_array("W",
                           shape=W_shape,
                           dtype=W.dtype,
                           storage=W.storage,
                           strides=W_strides,
                           transient=False)
        new_sdfg.add_array("X",
                           shape=X_shape,
                           dtype=X.dtype,
                           storage=X.storage,
                           strides=X_strides,
                           transient=False)
        new_sdfg.add_array("Y",
                           shape=Y_shape,
                           dtype=Y.dtype,
                           storage=Y.storage,
                           strides=Y_strides,
                           transient=False)
        if B is not None:
            new_sdfg.add_array("B",
                               shape=B_shape,
                               dtype=B.dtype,
                               storage=B.storage,
                               strides=B_strides,
                               transient=False)

        # GEMM Parameters: it is the output that is vectorized
        vec_width = Y.veclen

        # TODO: accept parametric?

        #if Y.veclen !=1 else math.gcd(16, output_size_x)
        #N = num_filters
        K = num_channels * filter_hx * filter_hy
        M = output_size_y * output_size_x * vec_width  # number of output elements (plain data, not vectorized)

        # Computational unit parameters:
        # - #PEs (optimal number is the number of filters)
        # - Tile size T (optimal number is M)
        P = 16 # Num PEs  #TODO parametric
        T = 128 # expressed in plain elements
        assert (T % vec_width == 0)


        #safe delay: we should wait 11 clock cycles before adding again into the same floating point number
        # stored in BRAM
        L = max(11 - T//vec_width, 0)

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
                    "n0": "0:ceiling({}/{})".format(num_filters, P),
                    "tm": "0:ceiling({}/{})".format(
                        M, T),  # must be repeated according to the tile size
                    "cin": "0:{}".format(num_channels),
                    "hx": "0:{}".format(filter_hx),
                    "hy": "0:{}".format(filter_hy)
                },
                schedule=dace.ScheduleType.FPGA_Device)

            # use a different map, and unroll it if necessary
            unroll_inner_map = P > (T//vec_width + L) and P <= 16
            send_map_entry, send_map_exit = state.add_map(
                "send_weights", {"n1": "0:{}".format(P)},
                schedule=dace.ScheduleType.FPGA_Device,
                unroll=unroll_inner_map)

            mem = state.add_read("W")
            pipe = state.add_write("W_pipe")
            tasklet = state.add_tasklet(
                "read_W", {"from_memory"}, {"to_kernel"}, """\
data = from_memory if n0*{}+ n1< {} else 0
to_kernel = data
""".format(P, num_filters))

            state.add_memlet_path(
                mem,
                entry,
                send_map_entry,
                tasklet,
                dst_conn="from_memory",
                memlet=dace.Memlet("W[n0 * {} + n1, cin, hx, hy]".format(P), dynamic=True, allow_oob=True))
            state.add_memlet_path(tasklet,
                                  send_map_exit,
                                  exit,
                                  pipe,
                                  src_conn="to_kernel",
                                  memlet=dace.Memlet(
                                      "W_pipe[{} -n1 -1]".format(P)))

        def make_read_im2col(state, sdfg, vec_width=1):

            # Matrix B will be the im2col matrix. We will build it row-by-row
            # to facilitate streaming in the systolic GEMM, avoiding storing it back to memory
            # Note: this will require to load multiple times the input feature, yet this save I/Os
            # The im2col matrix has size (num_channels * filter_hx * filter_hy) x (output_size_y * output_size_x)
            # We are using tiling along M (out_y*out_x)

            # gear boxing: we read plain data types, we stream vector data types
            # Therefore we have two maps, the innermost is unrolled
            im2col_me, im2col_mx = state.add_map(
                "im2col_map",
                {
                    "b": "0:{}".format(batch_size),
                    "n": "0:ceiling({}/{})".format(
                        num_filters, P),  # repeat B for computing the result
                    "tm": "0:ceiling({}/{})".format(
                        M, T),  # M already consider vec_width,
                    "cin": "0:{}".format(num_channels),
                    "hy": "0:{}".format(filter_hy),
                    "hx": "0:{}".format(
                        filter_hx),  #cin, hx and hy iterates over the row
                    # "xy": "0:{}/{}".format(output_size_x*output_size_y, vec_width)
                    "m0": "0:{}/{}".format(T, vec_width)
                    # "y": "0:{}".format(output_size_y),
                    # "x0": "0:{}/{}".format(output_size_x,
                    #                        vec_width),  #TODO vectorize read
                },
                schedule=dace.ScheduleType.FPGA_Device)

            read_map_entry, read_map_exit = state.add_map(
                "unrolled_reads_X", {"m1": "0:{}".format(vec_width)},
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
            tasklet = state.add_tasklet(
                "read_X", {"from_memory"}, {"to_kernel"}, """\
data = from_memory if  tm*{} + m0*{} + m1 < {} else 0
to_kernel = data
""".format(T, vec_width, M))
            # im2col_input_memlet = dace.Memlet(
            #         f"X[b, cin, xy*{vec_width}//{output_size_x} + hy, xy*{vec_width}%{output_size_x}+m1 + hx]")
            # im2col_input_memlet = dace.Memlet(
            #     "X[b, cin, y + hy, x0*{}+x1 + hx]".format(vec_width))
            # report everything to x, y: notice the integer division with int_floor
            im2col_input_memlet = dace.Memlet(
                f"X[b, cin, int_floor((tm*{T/vec_width} + m0),{output_size_x}) + hy, ((tm*{T/vec_width} + m0)%{output_size_x})*{vec_width} + m1 + hx]",
                dynamic=True)

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
                                  memlet=dace.Memlet("vec_data_im2col[m1]"))

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
            # Given the current systolic implementation, we will receive it one row at a time,
            # tile by tile

            # We don't need to accumulate on Y, but we need to add Biases (if present)

            pipe = state.add_read("Y_pipe")
            mem = state.add_write("Y")
            if add_bias is True:
                B = state.add_read("B")
            entry_map, exit_map = state.add_map(
                "write_Y",
                {
                    "b": "0:{}".format(batch_size),
                    "n0": "0:ceiling({}/{})".format(num_filters, P),
                    "tm": "0:ceiling({}/{})".format(M, T),
                    "n1": "0:{}".format(P),
                    "m": "0:{}/{}".format(T, vec_width)
                    # "y": "0:{}".format(output_size_y),
                    # "x": "0:{}".format(output_size_x)
                },
                schedule=dace.ScheduleType.FPGA_Device)

            # TODO: Xilinx: do we need to unroll bias addition?

            input_connectors = {"in_con"}
            if add_bias is True: input_connectors.add("bias")

            #in the copy tasklet, check that we are not out-of-range
            # and that we are receiving data from a PE that is doing useful work
            copy__add_bias__tasklet = state.add_tasklet(
                'copy_from_stream_Y', input_connectors, {'out_con'}, """\
if tm * {} + m * {} < {} and  n0*{}+n1 < {}: 
    out_con = in_con {}""".format(T, vec_width, M, P,  num_filters, "+ bias" if add_bias is True else ""))

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
                                      memlet=dace.Memlet("B[n0*{}+n1]".format(P), dynamic=True))
            # Memlet to memory

            # state.add_memlet_path(copy__add_bias__tasklet,
            #                       exit_map,
            #                       mem,
            #                       src_conn="out_con",
            #                       memlet=dace.Memlet("Y[b, n,  y, x]"))
            state.add_memlet_path(
                copy__add_bias__tasklet,
                exit_map,
                mem,
                src_conn="out_con",
                memlet=dace.Memlet(
                    f"Y[b, n0*{P}+n1,  int_floor(tm*{T} + m*{vec_width}, {output_size_y}), (tm*{T}+m*{vec_width})%{output_size_y}//{vec_width}]",
                    dynamic=True))

        def make_compute(sdfg, state, vec_width=1):
            vec_type = dace.vector(dace.float32, vec_width)
            W_pipe_in = state.add_read("W_pipe")
            im2col_pipe_in = state.add_read("im2col_pipe")
            im2col_pipe_out = state.add_write("im2col_pipe")
            Y_pipe_in = state.add_read("Y_pipe")
            Y_pipe_out = state.add_write("Y_pipe")

            # Safe delay for draining

            # Create a single pipeline

            entry_pipeline, exit_pipeline = state.add_pipeline(
                "compute_and_drain",
                {
                    "b": "0:{}".format(batch_size),
                    "n0": "0:ceiling({}/{})".format(num_filters, P), # P could not divide num_filters
                    "tm": "0:ceiling({}/{})".format(M, T),
                    "k": "0:{}".format(K),
                    "m": "0:{}/{} + {}".format(
                        T, vec_width, L
                    )  # The + L is a safe delay between computing and drain. It must be computed by
                    #considering the latency for updating the same result (not just the FP32 multiply add, but
                    # also for reading/writing
                },
                drain_size=P * T//vec_width,
                drain_overlap=False,
                additional_iterators={
                    'm_drain': 0,
                    'k_drain': 0
                },
                schedule=dace.ScheduleType.FPGA_Device)

            # Instantiate buffers
            sdfg.add_scalar("W_reg",
                            dtype=dace.float32,
                            transient=True,
                            storage=dace.dtypes.StorageType.FPGA_Registers)
            W_reg_init = state.add_access("W_reg")
            W_reg = state.add_access("W_reg")

            # Note: for some of the Sacred Mysteries of Intel OpenCL Compiler (TM), if this buffer is smaller
            # than 24 floats, the II of the pipeline will be 5. Therefore we check this (with 32 to be
            # more compliant with standard vector size) and in case we enlarge it

            buffer_size = max(T, 32) / vec_width
            # For C result we are going to use vectorized data type
            sdfg.add_array(
                "Y_buffer",
                [buffer_size],  #M already accounts for vec width
                dtype=vec_type,
                transient=True,
                storage=dace.dtypes.StorageType.FPGA_Local)
            Y_buffer_in = state.add_read("Y_buffer")
            Y_buffer_out = state.add_write("Y_buffer")

            # Buffering of im2col data (B)
            sdfg.add_array("im2col_reg",
                           shape=[1],
                           dtype=vec_type,
                           transient=True,
                           storage=dace.dtypes.StorageType.FPGA_Local)
            im2col_reg = state.add_access("im2col_reg")

            # every PE: reads input data, buffer the data assigned to it, forwards the data
            buffer_w_tasklet = state.add_tasklet(
                "buffer_w", {"w_in"}, {"w_reg"}, """\
if m == 0 and not {}:
    w_reg = w_in""".format(entry_pipeline.pipeline.drain_condition()))
            state.add_memlet_path(W_pipe_in,
                                  entry_pipeline,
                                  buffer_w_tasklet,
                                  memlet=dace.Memlet("W_pipe[p]",
                                                     dynamic=True),
                                  dst_conn="w_in")
            state.add_memlet_path(buffer_w_tasklet,
                                  W_reg,
                                  memlet=dace.Memlet("W_reg[0]", dynamic=True),
                                  src_conn="w_reg")

            # FEED B (im2col matrix)
            # Read B: done outside of the compute tasklet to help type inference

            buffer_im2col_tasklet = state.add_tasklet(
                "buffer_im2col", {"im2col_in"}, {"im2col_reg_out"}, """\
if  m>={} and not {}:
    im2col_reg_out = im2col_in""".format(
                    L, entry_pipeline.pipeline.drain_condition()))

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
                                  src_conn="im2col_reg_out")

            # COMPUTE AND DRAIN
            # Compute and forward B: this is done if we are not in the init phase of the pipeline
            compute_tasklet = state.add_tasklet(
                "compute_and_drain",
                {"w_in", "im2col_in", "y_in", "forward_in"},
                {"im2col_out", "y_out", "y_pipe_out"}, f"""\
if m>= {L} and not {entry_pipeline.pipeline.drain_condition()}:
    y_prev = 0 if k == 0 else y_in     
    y_out =  y_prev + w_in * im2col_in
    if p < {P} - 1:
        im2col_out = im2col_in
# Drain
# when we have to drain:
# - if k = K-1 and m>=L: drain my own result
#-  otherwise, if k_drain<p forward data coming from previous PEs (this could happens also in the drain phase)
if((b>0  or n0 > 0 or tm > 0)  and k_drain <p and m_drain <{T}/{vec_width}) or  (k=={K}-1 and m>= {L}) or ({entry_pipeline.pipeline.drain_condition()} and k_drain < p):
    y_pipe_out = y_out if (p==0 or (k_drain=={K}-1 and not {entry_pipeline.pipeline.drain_condition()})) else forward_in

# adjust draining iterators
if not {entry_pipeline.pipeline.drain_condition()}:
    if m_drain >= {L} +  {T}/{vec_width} -1:
        m_drain = 0
        if k_drain >= {K} - 1:
            k_drain = 0
        else:
            k_drain = k_drain +1
    else:
        m_drain = m_drain + 1
else:
    if m_drain >=  {T}/{vec_width} -1:
        m_drain = 0
        if k_drain >= {K} - 1:
            k_drain = 0
        else:
            k_drain = k_drain +1
    else:
        m_drain = m_drain + 1
""")

            state.add_memlet_path(W_reg,
                                  compute_tasklet,
                                  dst_conn="w_in",
                                  memlet=dace.Memlet("W_reg[0]"))
            state.add_memlet_path(im2col_reg,
                                  compute_tasklet,
                                  memlet=dace.Memlet("im2col_reg[p]",
                                                     dynamic=False),
                                  dst_conn="im2col_in")
            state.add_memlet_path(compute_tasklet,
                                  exit_pipeline,
                                  im2col_pipe_out,
                                  memlet=dace.Memlet("im2col_pipe[p + 1]",
                                                     dynamic=True),
                                  src_conn="im2col_out")
            state.add_memlet_path(Y_buffer_in,
                                  entry_pipeline,
                                  compute_tasklet,
                                  dst_conn="y_in",
                                  memlet=dace.Memlet(
                                      "Y_buffer[m-{}]".format(L),
                                      allow_oob=True))
            state.add_memlet_path(compute_tasklet,
                                  exit_pipeline,
                                  Y_buffer_out,
                                  src_conn="y_out",
                                  memlet=dace.Memlet(
                                      "Y_buffer[m-{}]".format(L),
                                      allow_oob=True,
                                      dynamic=True))

            state.add_memlet_path(Y_pipe_in,
                                  entry_pipeline,
                                  compute_tasklet,
                                  memlet=dace.Memlet("Y_pipe[p-1]",
                                                     dynamic=True),
                                  dst_conn="forward_in")
            state.add_memlet_path(compute_tasklet,
                                  exit_pipeline,
                                  Y_pipe_out,
                                  memlet=dace.Memlet("Y_pipe[p]",
                                                     dynamic=True),
                                  src_conn="y_pipe_out")

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
            # state.add_memlet_path(W_pipe_out,
            #                       compute_exit,
            #                       memlet=dace.memlet.Memlet())
            state.add_memlet_path(im2col_pipe_out,
                                  compute_exit,
                                  memlet=dace.memlet.Memlet())
            state.add_memlet_path(Y_pipe_out,
                                  compute_exit,
                                  memlet=dace.memlet.Memlet())
            state.add_memlet_path(compute_entry,
                                  W_reg_init,
                                  memlet=dace.memlet.Memlet())
            state.add_memlet_path(W_reg_init,
                                  entry_pipeline,
                                  memlet=dace.memlet.Memlet())
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

        # build the compute State
        vec_type = dace.vector(dace.float32, vec_width)

        new_sdfg.add_stream("W_pipe",
                            dace.float32,
                            transient=True,
                            shape=(P, ),
                            storage=dace.dtypes.StorageType.FPGA_Local,
                            buffer_size=str(P))
        new_sdfg.add_stream("im2col_pipe",
                            vec_type,
                            transient=True,
                            shape=(P + 1, ),
                            buffer_size=2,
                            storage=dace.dtypes.StorageType.FPGA_Local)
        new_sdfg.add_stream("Y_pipe",
                            vec_type,
                            transient=True,
                            shape=(P + 1, ),
                            buffer_size=T,
                            storage=dace.dtypes.StorageType.FPGA_Local)

        make_read_W(new_state)
        make_read_im2col(new_state, new_sdfg, vec_width)
        make_compute(new_sdfg, new_state, vec_width)
        make_write_Y(new_state, new_sdfg, vec_width, add_bias=(B is not None))

        new_sdfg.fill_scope_connectors()
        # Specialize the new sdfg, by using the input shapes
        new_sdfg.save("/tmp/conv.sdfg")
        new_sdfg.validate()
        # nest the sdfg and map symbols
        expansion = state.add_nested_sdfg(new_sdfg,
                                          sdfg,
                                          node.in_connectors,
                                          node.out_connectors,
                                          name=node.name,
                                          debuginfo=node.debuginfo,
                                          symbol_mapping=symbol_mapping)
        expansion.sdfg.save('/tmp/expanded.sdfg')
        return expansion
