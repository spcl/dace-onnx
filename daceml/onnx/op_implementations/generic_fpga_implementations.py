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


@autoregister_params(op="Relu", name="generic_fpga")
class FPGARelu(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        X = in_desc_with_name(node, state, sdfg, "X")
        Y = out_desc_with_name(node, state, sdfg, "Y")

        # TODO deal with this. Right Now I'm doing it to
        # gently introduce streaming
        vec_width = X.veclen

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
        return new_sdfg



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
                 c="0:{}".format(num_channels),
                 out_y="0:{}".format(output_size_y),
                 out_x="0:{}".format(output_size_x)))


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
            "if hy == {} - 1 and hx == {} -1: output = max_out"
            .format(dtypes.min_value(Y.dtype), filter_height, filter_width,
                    filter_height, filter_height, vec_width, filter_height,
                    filter_width))


        read_X = new_state.add_read("X")
        write_Y = new_state.add_write("Y")
        read_max_res = new_state.add_access("max_res")
        write_max_res = new_state.add_write("max_res")

        # memlets: input data


        new_state.add_memlet_path(read_X,
                                  outer_me,
                                  inner_me,
                                  compute_tasklet,
                                  dst_conn="image_in",
                                  memlet=dace.Memlet("X[b, c, out_y*{}+hy, out_x*{}+hx]".format(filter_height, filter_height)))
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

        new_sdfg.fill_scope_connectors()
        new_sdfg.save("/tmp/maxpool.sdfg")
        return new_sdfg