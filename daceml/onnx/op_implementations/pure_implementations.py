import copy
import itertools
import logging
import typing

import dace
import numpy as np
from dace import SDFGState, SDFG, nodes
from dace.frontend.common import create_einsum_sdfg
from dace.sdfg.nodes import Node

from daceml.onnx import converters
from daceml.onnx.forward_implementation_abc import ONNXForward
from daceml.onnx.nodes import onnx_op
from daceml.onnx.op_implementations.utils import op_implementation, program_for_node
from daceml.transformation import constant_folding
from daceml.util.utils import in_desc_with_name, out_desc_with_name, in_edge_with_name, iterables_equal

log = logging.getLogger(__name__)


@op_implementation(op="Log", name="pure")
class PureLog(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return in_desc_with_name(node, state, sdfg, 'input').dtype in [
            dace.float16, dace.float32, dace.float64
        ]

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        def prog(input, output):
            output[:] = dace.elementwise(lambda x: log(x), input)

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="Sqrt", name="pure")
class PureSqrt(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return in_desc_with_name(node, state, sdfg, 'X').dtype in [
            dace.float16, dace.float32, dace.float64
        ]

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        def prog(X, Y):
            Y[:] = dace.elementwise(lambda x: sqrt(x), X)

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="Pow", name="pure")
class PurePow(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return in_desc_with_name(node, state, sdfg, 'X').dtype in [
            dace.float16, dace.float32, dace.float64
        ]

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        def prog(X, Y, Z):
            Z[:] = X**Y

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="Add", name="pure")
class PureAdd(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        def prog(A, B, C):
            C[:] = A + B

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="Sub", name="pure")
class PureSub(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        def prog(A, B, C):
            C[:] = A - B

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="Mul", name="pure")
class PureMul(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        def prog(A, B, C):
            C[:] = A * B

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="Div", name="pure")
class PureDiv(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        def prog(A, B, C):
            C[:] = A / B

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="ReduceMean", name="pure")
class PureReduceMean(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        axes = node.axes

        # when keepdims is true, this works but there is a useless copy. We just leave this for now; this can be fixed
        # with a reshape node when those exist.
        def prog(data, reduced):
            reduced[:] = np.mean(data, axis=axes)

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="Erf", name="pure")
class PureErf(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return in_desc_with_name(node, state, sdfg, 'input').dtype in [
            dace.float16, dace.float32, dace.float64
        ]

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        def prog(input, output):
            output[:] = dace.elementwise(lambda x: erf(x), input)

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="MatMul", name="pure")
class PureMatMul(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        input0_dim = len(in_desc_with_name(node, state, sdfg, "A").shape)
        input1_dim = len(in_desc_with_name(node, state, sdfg, "B").shape)

        if input0_dim == 1 or input1_dim == 1:
            return False
        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        A_desc = in_desc_with_name(node, state, sdfg, "A")
        B_desc = in_desc_with_name(node, state, sdfg, "B")
        Y_desc = out_desc_with_name(node, state, sdfg, "Y")
        input0_dim = A_desc.shape
        input1_dim = B_desc.shape

        # list containing letters from z-a
        letters = [chr(ord('z') - i) for i in range(26)]
        # i j k are used for the last dimensions
        letters = [l for l in letters if l not in ['i', 'j', 'k']]

        if len(input0_dim) == 1:
            if len(input1_dim) != 2:
                raise ValueError("invalid dimensions")
            arg1 = 'k'
            arg2 = 'kj'
            result = 'j'
        elif len(input1_dim) == 1:
            if len(input0_dim) != 2:
                raise ValueError("invalid dimensions")
            arg1 = 'ik'
            arg2 = 'k'
            result = 'i'
        else:
            # build the einsum. The last two dimensions are always just the matrix multiply einsum
            # dace will later specialize to a batched matmul if possible
            arg1 = 'ik'
            arg2 = 'kj'
            result = 'ij'
            if input0_dim[-2] != input0_dim[-1]:
                if dace.symbolic.issymbolic(input0_dim[-2]):
                    log.warning(
                        f"overriding symbol {input0_dim[-2]} with value {input1_dim[-1]} in descriptor of input A of node {node}"
                    )
                    new_shape = list(A_desc.shape)
                    new_shape[-1] = input1_dim[-2]
                    A_desc.shape = new_shape
                elif dace.symbolic.issymbolic(input1_dim[-1]):
                    log.warning(
                        f"overriding symbol {input0_dim[-1]} with value {input0_dim[-2]} in descriptor of input B of node {node}"
                    )
                    new_shape = list(B_desc.shape)
                    new_shape[-2] = input0_dim[-1]
                    B_desc.shape = new_shape
            input0_dim = input0_dim[:-2]
            input1_dim = input1_dim[:-2]
            for dim0, dim1 in itertools.zip_longest(reversed(input0_dim),
                                                    reversed(input1_dim)):
                if dim0 is None:
                    # only dim0 exists
                    letter = letters.pop()
                    arg2 = letter + arg2
                    result = letter + result
                elif dim1 is None:
                    # only dim1 exists
                    letter = letters.pop()
                    arg1 = letter + arg1
                    result = letter + result
                else:
                    # both exist
                    letter = letters.pop()
                    arg1 = letter + arg1
                    arg2 = letter + arg2
                    result = letter + result

        einsum_str = '{},{}->{}'.format(arg1, arg2, result)

        # we lower to an ONNXEinsum node instead straight to the dace einsum to make the autodiff simpler
        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()
        einsum_node: nodes.LibraryNode = onnx_op.ONNXEinsum(
            node.label + "_einsum_expansion", equation=einsum_str)

        nstate.add_node(einsum_node)
        einsum_node.add_in_connector("Inputs__0")
        einsum_node.add_in_connector("Inputs__1")
        nsdfg.add_datadesc("A", copy.deepcopy(A_desc))
        nsdfg.add_datadesc("B", copy.deepcopy(B_desc))
        nsdfg.add_datadesc("Y", copy.deepcopy(Y_desc))
        nsdfg.arrays["A"].transient = False
        nsdfg.arrays["B"].transient = False
        nsdfg.arrays["Y"].transient = False

        nstate.add_edge(nstate.add_read("A"), None, einsum_node, "Inputs__0",
                        nsdfg.make_array_memlet("A"))
        nstate.add_edge(nstate.add_read("B"), None, einsum_node, "Inputs__1",
                        nsdfg.make_array_memlet("B"))
        nstate.add_edge(einsum_node, "Output", nstate.add_write("Y"), None,
                        nsdfg.make_array_memlet("Y"))

        return nsdfg


@op_implementation(op="Einsum", name="pure")
class PureEinsum(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        if "..." in node.equation:
            return False
        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        nsdfg = dace.SDFG(node.label + "_expansion")
        nstate = nsdfg.add_state()

        for e in node.iter_inputs_in_onnx_order(state):
            nsdfg.add_datadesc(
                e.dst_conn, in_desc_with_name(node, state, sdfg, e.dst_conn))
        for e in node.iter_outputs_in_onnx_order(state):
            nsdfg.add_datadesc(
                e.src_conn, out_desc_with_name(node, state, sdfg, e.src_conn))

        create_einsum_sdfg(None,
                           nsdfg,
                           nstate,
                           node.equation.replace(" ", ""),
                           *(e.dst_conn
                             for e in node.iter_inputs_in_onnx_order(state)),
                           output="Output")
        return nsdfg


@op_implementation(op="Identity", name="pure")
class PureIdentity(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        def prog(input, output):
            output[:] = input

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="Reciprocal", name="pure")
class PureReciprocal(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        return in_desc_with_name(node, state, sdfg, 'X').dtype in [
            dace.float16, dace.float32, dace.float64
        ]

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        dtype = in_desc_with_name(node, state, sdfg, 'X').dtype
        tanh_lambda = "lambda x: dace.{}(1) / x".format(dtype.to_string())

        def prog(X, Y):
            Y[:] = dace.elementwise(tanh_lambda, X)

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="Tanh", name="pure")
class PureTanh(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        def prog(input, output):
            output[:] = dace.elementwise(lambda x: tanh(x), input)

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="ReduceSum", name="pure")
class PureReduceSum(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        axes = node.axes

        # when keepdims is true, this works but there is a useless copy. We just leave this for now; this can be fixed
        # with a reshape node when those exist.
        def prog(data, reduced):
            reduced[:] = np.sum(data, axis=axes)

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="ReduceMax", name="pure")
class PureReduceMax(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        axes = node.axes

        # when keepdims is true, this works but there is a useless copy. We just leave this for now; this can be fixed
        # with a reshape node when those exist.
        def prog(data, reduced):
            reduced[:] = np.max(data, axis=axes)

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="ReduceMin", name="pure")
class PureReduceMin(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        axes = node.axes

        # when keepdims is true, this works but there is a useless copy. We just leave this for now; this can be fixed
        # with a reshape node when those exist.
        def prog(data, reduced):
            reduced[:] = np.min(data, axis=axes)

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="Softmax", name="pure")
class PureSoftmax(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        axis = node.axis

        reduced_shape = list(
            copy.deepcopy(in_desc_with_name(node, state, sdfg, "input").shape))
        reduced_shape[axis] = 1

        def prog(input, output):
            max = np.max(input, axis=axis)
            max_keepdims = np.reshape(max, reduced_shape)
            exp_arr = np.exp(input - max_keepdims)
            sum = np.sum(exp_arr, axis=axis)
            sum_keepdims = np.reshape(sum, reduced_shape)
            output[:] = exp_arr / sum_keepdims

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="Transpose", name="pure")
class PureTranspose(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        perm = node.perm

        def prog(data, transposed):
            transposed[:] = np.transpose(data, axes=perm)

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="Cast", name="pure")
class PureCast(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:

        if (in_desc_with_name(node, state, sdfg,
                              "input").dtype == out_desc_with_name(
                                  node, state, sdfg, "output").dtype):
            return True

        target_type = node.to
        try:
            converters.onnx_tensor_type_to_typeclass(target_type)
        except ValueError:
            return False

        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        input_desc = in_desc_with_name(node, state, sdfg, "input")
        output_desc = out_desc_with_name(node, state, sdfg, "output")
        if (input_desc.dtype == output_desc.dtype):

            def prog(input, output):
                output[:] = input
        else:

            def prog(input, output):
                output[:] = dace.elementwise(lambda x: x, input)

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="Gemm", name="pure")
class PureGemm(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        if node.alpha == 1.0 and node.beta == 1.0 and node.transA == 0 and node.transB == 1:
            return True
        return False

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        assert node.alpha == 1.0 and node.beta == 1.0 and node.transA == 0 and node.transB == 1

        # the gemm libnode is broken for now, so we just do it manually
        if "C" in node.in_connectors:

            def prog(A, B, C, Y):
                Y[:] = A @ np.transpose(B) + C
        else:

            def prog(A, B, Y):
                Y[:] = A @ np.transpose(B)

        sdfg = program_for_node(prog, sdfg, state, node)
        sdfg.apply_strict_transformations()
        return sdfg


@op_implementation(op="Relu", name="pure")
class PureRelu(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:
        input_dtype = in_desc_with_name(node, state, sdfg, "X").dtype
        cast_lambda = "lambda x: max(x, dace.{}(0))".format(
            input_dtype.to_string())

        def prog(X, Y):
            Y[:] = dace.elementwise(cast_lambda, X)

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="Reshape", name="pure")
class PureReshape(ONNXForward):
    '''
        Reshape expansion: this relies on views
    '''
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        input_name = "data"
        output_name = "reshaped"
        flatten = False

        # if called from Flatten
        if "input" in node._in_connectors.keys():
            input_name = "input"
            output_name = "output"
            flatten = True

        new_shape = out_desc_with_name(node, state, sdfg, output_name).shape

        if not flatten:
            node.remove_in_connector("shape")
            shape_node = in_edge_with_name(node, state, "shape").src
            constant_folding.remove_node_and_computation(
                sdfg, state, shape_node)

        if not flatten:

            def prog(data, reshaped):
                reshaped[:] = np.reshape(data, new_shape)
        else:

            def prog(input, output):
                output[:] = np.reshape(input, new_shape)

        return program_for_node(prog, sdfg, state, node)


@op_implementation(op="Flatten", name="pure")
class PureFlatten(ONNXForward):
    '''
        Flatten Expansion, reuses Reshape implementation
    '''
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        # Reuse Reshape implementation
        return PureReshape.forward(node, state, sdfg)


@op_implementation(op="Sum", name="pure")
class PureSum(ONNXForward):
    @staticmethod
    def forward_can_be_applied(node: onnx_op.ONNXOp, state: SDFGState,
                               sdfg: SDFG) -> bool:
        # check that all shapes are arrays, and that the shapes are all equal
        shape = None
        for edge in node.iter_inputs_in_onnx_order(state):
            desc = in_desc_with_name(node, state, sdfg, edge.dst_conn)
            if shape is None:
                shape = desc.shape

            if not iterables_equal(shape, desc.shape):
                return False

        if not iterables_equal(
                shape,
                out_desc_with_name(node, state, sdfg, "sum").shape):
            return False

        return True

    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[Node, SDFG]:

        nsdfg = dace.SDFG(node.name)
        input_names = []
        for e in node.iter_inputs_in_onnx_order(state):
            new_desc = copy.deepcopy(
                in_desc_with_name(node, state, sdfg, e.dst_conn))
            new_desc.transient = False
            nsdfg.add_datadesc(e.dst_conn, new_desc)
            input_names.append(e.dst_conn)

        new_desc = copy.deepcopy(out_desc_with_name(node, state, sdfg, "sum"))
        new_desc.transient = False
        nsdfg.add_datadesc("sum", new_desc)

        nstate = nsdfg.add_state()
        # we know all shapes are equal to the output shape
        shape = out_desc_with_name(node, state, sdfg, "sum").shape
        map_ranges = {f"i{i}": f"0:{s}" for i, s in enumerate(shape)}
        index_str = f"{', '.join(map_ranges.keys())}"
        tasklet, _, _ = nstate.add_mapped_tasklet(
            node.name + "_tasklet",
            map_ranges=map_ranges,
            inputs={
                f"__{inp}": dace.Memlet(f"{inp}[{index_str}]")
                for inp in input_names
            },
            code=f"__sum = {' + '.join(f'__{inp}' for inp in input_names)}",
            outputs={"__sum": dace.Memlet(f"sum[{index_str}]")},
            external_edges=True)

        tasklet.in_connectors = {
            f"__{inp}": in_desc_with_name(node, state, sdfg, inp).dtype
            for inp in input_names
        }
        tasklet.out_connectors = {
            "__sum": out_desc_with_name(node, state, sdfg, "sum").dtype
        }
        return nsdfg


@op_implementation(op="LogSoftmax", name="pure")
class PureLogSoftmax(ONNXForward):
    @staticmethod
    def forward(node: onnx_op.ONNXOp, state: SDFGState,
                sdfg: SDFG) -> typing.Union[nodes.Node, SDFG]:

        axis = node.axis

        reduced_shape = list(
            copy.deepcopy(in_desc_with_name(node, state, sdfg, "input").shape))
        reduced_shape[axis] = 1

        def prog(input, output):
            max = np.max(input, axis=axis)
            max_keepdims = np.reshape(max, reduced_shape)
            max_sub = input - max_keepdims
            exp_arr = np.exp(max_sub)
            sum = np.sum(exp_arr, axis=axis)
            sum_keepdims = np.reshape(sum, reduced_shape)
            log_sum = np.log(sum_keepdims)
            output[:] = max_sub - log_sum

        return program_for_node(prog, sdfg, state, node)
