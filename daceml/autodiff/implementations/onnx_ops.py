import copy
import itertools
import typing
from typing import List, Optional, Tuple, Dict, Union

import dace
from dace.frontend.common import einsum
from dace.registry import autoregister_params
from dace import nodes as nd, dtypes

import daceml.onnx as donnx
from daceml.onnx.op_implementations import pure_implementations, cudnn_implementations, empty_sdfg_for_node, \
    clean_onnx_name, environments
import daceml.autodiff.utils as butils
from daceml.autodiff.base_abc import BackwardImplementation, BackwardContext, BackwardResult


def reverse_einsum_wrt_input(forward_node: donnx.ONNXEinsum,
                             input_name: str) -> Tuple[List[str], str]:
    """ Produce the einsum string that computes the grad of ``forward_node`` w.r.t. ``input_name``.

       :Note:
            There is an edge case we currently don't handle (can be implemented though). Something like ``'ii->i'``
            would become ``'i->ii'``. This is invalid because ``i`` is repeated in the output.

        :param forward_node: the einsum node to reverse.
        :param input_name: the connector on the forward node the produce the gradient computation for.
        :return: the list of forward node connectors required as inputs, and the einsum string. The first parameter of
                 the produced einsum string is implicitly the grad of ``Output``.
    """

    _, input_idx = donnx.parse_variadic_param(input_name)
    parser = einsum.EinsumParser(forward_node.equation)

    backward_input_expressions = [
        parser.output
    ] + parser.inputs[:input_idx] + parser.inputs[input_idx + 1:]
    backward_input_arrays = [
        f"Inputs__{i}" for i in itertools.chain(
            range(input_idx), range(input_idx + 1, len(parser.inputs)))
    ]

    einsum_str = f"{','.join(backward_input_expressions)}->{parser.inputs[input_idx]}"
    return backward_input_arrays, einsum_str


@autoregister_params(op="Einsum", name="default")
class DefaultEinsumBackward(BackwardImplementation):
    """ The symbolic autodiff can automatically derive matmuls, but the produced maps are more difficult to optimize.
    """
    @staticmethod
    def backward_can_be_applied(node: nd.Node, state: dace.SDFGState,
                                sdfg: dace.SDFG) -> bool:
        return pure_implementations.PureEinsum.forward_can_be_applied(
            node, state, sdfg)

    @staticmethod
    def backward(
        forward_node: nd.Node, context: BackwardContext,
        given_gradients: List[Optional[str]],
        required_gradients: List[Optional[str]]
    ) -> Tuple[nd.Node, BackwardResult]:

        nsdfg = dace.SDFG(forward_node.label + "_backward")
        nstate = nsdfg.add_state()

        # setup arrays
        output_desc = butils.forward_out_desc_with_name(
            forward_node, context, "Output")
        result = BackwardResult.empty()
        result.given_grad_names["Output"] = butils.add_backward_desc(
            nsdfg, context.forward_sdfg, output_desc, "Output")
        access_output_grad = nstate.add_read(result.given_grad_names["Output"])

        def create_access_node(connector: str) -> nd.AccessNode:
            nsdfg.add_datadesc(
                connector,
                butils.forward_in_desc_with_name(forward_node, context,
                                                 connector))
            return nstate.add_read(connector)

        # the forward inputs we will require
        # maps the connector name to the accessnode
        required_forward_inputs: Dict[str, nd.AccessNode] = {}

        for input_name in required_gradients:
            # we add an einsum for each required gradient
            forward_inputs, einsum_str = reverse_einsum_wrt_input(
                forward_node, input_name)

            einsum_node = donnx.ONNXEinsum(input_name + "_backward",
                                           equation=einsum_str)
            nstate.add_node(einsum_node)

            # the first input is always the output grad
            einsum_node.add_in_connector(f"Inputs__0")
            nstate.add_edge(
                access_output_grad, None, einsum_node, "Inputs__0",
                nsdfg.make_array_memlet(result.given_grad_names["Output"]))

            # add the other inputs from forward that we need
            for i, forward_input in enumerate(forward_inputs):
                connector = f"Inputs__{i + 1}"
                einsum_node.add_in_connector(connector)
                if forward_input not in required_forward_inputs:
                    required_forward_inputs[
                        forward_input] = create_access_node(forward_input)

                nstate.add_edge(required_forward_inputs[forward_input], None,
                                einsum_node, connector,
                                nsdfg.make_array_memlet(forward_input))

            # write out the gradient
            butils.forward_in_desc_with_name(forward_node, context, input_name)
            result.required_grad_names[
                input_name] = butils.add_backward_desc_for_connector(
                    nsdfg, forward_node, context, input_name, True)
            memlet = nsdfg.make_array_memlet(
                result.required_grad_names[input_name])
            nstate.add_edge(
                einsum_node, "Output",
                nstate.add_write(result.required_grad_names[input_name]), None,
                memlet)

        result_node = context.backward_state.add_nested_sdfg(
            nsdfg, None,
            set(result.given_grad_names.values()).union(
                required_forward_inputs),
            set(result.required_grad_names.values()))

        return result_node, result


@autoregister_params(op="Softmax", name="default")
class DefaultSoftmaxBackward(BackwardImplementation):
    @staticmethod
    def backward(
        forward_node: nd.Node, context: BackwardContext,
        given_gradients: List[Optional[str]],
        required_gradients: List[Optional[str]]
    ) -> Tuple[Union[nd.Node, dace.SDFG], BackwardResult]:

        dim = forward_node.axis

        output_shape = butils.forward_out_desc_with_name(
            forward_node, context, "output").shape
        output_dtype = butils.forward_out_desc_with_name(
            forward_node, context, "output").dtype

        sums_shape = list(copy.deepcopy(output_shape))
        sums_shape[dim] = 1

        def softmax_backward(output, output_grad, input_grad):
            prod = dace.define_local(output_shape, output_dtype)
            sums = dace.define_local(sums_shape, output_dtype)
            donnx.ONNXMul(A=output, B=output_grad, C=prod)
            donnx.ONNXReduceSum(data=prod,
                                reduced=sums,
                                keepdims=1,
                                axes=[dim])

            donnx.ONNXMul(A=output, B=sums, C=input_grad)
            # let's not use ONNXSub here; not sure how this inplace op is handled by ORT...
            input_grad[:] = prod - input_grad

        result_node, result = butils.backward_program_for_node(
            softmax_backward, context, forward_node)

        butils.connect_output_from_forward(forward_node, result_node, context,
                                           "output")

        return result_node, result


@autoregister_params(op="LogSoftmax", name="default")
class DefaultLogSoftmaxBackward(BackwardImplementation):
    @staticmethod
    def backward(
        forward_node: nd.Node, context: BackwardContext,
        given_gradients: List[Optional[str]],
        required_gradients: List[Optional[str]]
    ) -> Tuple[nd.Node, BackwardResult]:

        dim = forward_node.axis
        output_shape = butils.forward_out_desc_with_name(
            forward_node, context, "output").shape
        output_dtype = butils.forward_out_desc_with_name(
            forward_node, context, "output").dtype

        sums_shape = list(copy.deepcopy(output_shape))
        sums_shape[dim] = 1

        def logsoftmax_backward(output, output_grad, input_grad):
            exp_output = dace.define_local(output_shape, output_dtype)
            donnx.ONNXExp(input=output, output=exp_output)

            grad_output_sum = dace.define_local(sums_shape, output_dtype)
            donnx.ONNXReduceSum(data=output_grad,
                                reduced=grad_output_sum,
                                keepdims=1,
                                axes=[dim])
            # let's not use ONNXMul here; not sure how this inplace op is handled by ORT...
            exp_output[:] = exp_output * grad_output_sum
            donnx.ONNXSub(A=output_grad, B=exp_output, C=input_grad)

        result_node, result = butils.backward_program_for_node(
            logsoftmax_backward, context, forward_node)

        butils.connect_output_from_forward(forward_node, result_node, context,
                                           "output")
        return result_node, result


@autoregister_params(op="Relu", name="pure")
class PureReluBackward(BackwardImplementation):
    @staticmethod
    def backward(
        forward_node: nd.Node, context: BackwardContext,
        given_gradients: List[Optional[str]],
        required_gradients: List[Optional[str]]
    ) -> Tuple[nd.Node, BackwardResult]:
        input_desc = butils.forward_in_desc_with_name(forward_node, context,
                                                      "X")

        new_sdfg = dace.SDFG(forward_node.label + "_backward")

        # setup arrays
        result = BackwardResult.empty()
        result.required_grad_names["X"] = butils.add_backward_desc(
            new_sdfg, context.forward_sdfg, input_desc, "X")
        result.given_grad_names["Y"] = butils.add_backward_desc(
            new_sdfg, context.forward_sdfg, input_desc, "Y")
        new_X_desc = copy.deepcopy(input_desc)
        new_X_desc.transient = False
        new_sdfg.add_datadesc("X", new_X_desc)

        # setup state
        new_state = new_sdfg.add_state()

        enum_shapes = list(enumerate(input_desc.shape))
        all_indices = ", ".join("__i{}".format(i) for i, _ in enum_shapes)

        # yapf: disable
        new_state.add_mapped_tasklet(
            "_relu_backward_",
            {
                "__i{}".format(i): "0:{}".format(s) for i, s in enum_shapes
            },
            {
                "__y_grad": dace.Memlet("Y_grad[{}]".format(all_indices)),
                "__x": dace.Memlet("X[{}]".format(all_indices))
            },
            "__x_grad = __y_grad if __x > dace.{0}(0) else dace.{0}(0)".format(
                input_desc.dtype.to_string()),
            {
                "__x_grad": dace.Memlet("X_grad[{}]".format(all_indices))
            },
            external_edges=True)
        # yapf: enable

        node = context.backward_state.add_nested_sdfg(new_sdfg, None,
                                                      {"Y_grad", "X"},
                                                      {"X_grad"})
        return node, result


@autoregister_params(op="BatchNormalization", name="cuDNN")
class CuDNNBatchNormBackward(BackwardImplementation):
    @staticmethod
    def backward_can_be_applied(node: nd.Node, state: dace.SDFGState,
                                sdfg: dace.SDFG) -> bool:
        return cudnn_implementations.CudnnBatchNormalizationTraining.forward_can_be_applied(
            node, state, sdfg)

    @staticmethod
    def backward(
        forward_node: nd.Node, context: BackwardContext,
        given_gradients: typing.List[typing.Optional[str]],
        required_gradients: typing.List[typing.Optional[str]]
    ) -> typing.Tuple[nd.Node, BackwardResult]:

        nsdfg = dace.SDFG(forward_node.label + "_backward")
        X_desc = butils.forward_in_desc_with_name(forward_node, context, "X")
        scale_desc = butils.forward_in_desc_with_name(forward_node, context,
                                                      "scale")
        T = X_desc.dtype

        # setup arrays
        result = BackwardResult.empty()
        result.required_grad_names["X"] = butils.add_backward_desc(
            nsdfg, context.forward_sdfg, X_desc, "X")
        result.given_grad_names["Y"] = butils.add_backward_desc_for_connector(
            nsdfg, forward_node, context, "Y", input=False)
        result.required_grad_names[
            "scale"] = butils.add_backward_desc_for_connector(nsdfg,
                                                              forward_node,
                                                              context,
                                                              "scale",
                                                              input=True)
        result.required_grad_names[
            "B"] = butils.add_backward_desc_for_connector(nsdfg,
                                                          forward_node,
                                                          context,
                                                          "B",
                                                          input=True)

        # input X
        new_X_desc = copy.deepcopy(X_desc)
        new_X_desc.transient = False
        nsdfg.add_datadesc("X", new_X_desc)

        # input scale
        new_scale_desc = copy.deepcopy(scale_desc)
        new_scale_desc.transient = False
        nsdfg.add_datadesc("scale", new_scale_desc)

        # setup state
        nstate = nsdfg.add_state()
        fwd_unique_id = "{}_{}_{}_{}".format(
            clean_onnx_name(forward_node.name), context.forward_sdfg.sdfg_id,
            context.forward_sdfg.node_id(context.forward_state),
            context.forward_state.node_id(forward_node))

        unique_id = f"{fwd_unique_id}_bwd"

        class Environment:
            cmake_minimum_version = None
            cmake_packages = []
            cmake_variables = {}
            cmake_includes = []
            cmake_libraries = []
            cmake_compile_flags = []
            cmake_link_flags = []
            cmake_files = []
            state_fields = [
                f"cudnnTensorDescriptor_t *{unique_id}_X_desc;",
                f"cudnnTensorDescriptor_t *{unique_id}_dX_desc;",
                f"cudnnTensorDescriptor_t *{unique_id}_dY_desc;",
                f"cudnnTensorDescriptor_t *{unique_id}_dScale_desc;",
            ]
            dependencies = [environments.cuDNN]
            headers = []
            init_code = ""
            finalize_code = ""

        Environment.__name__ = unique_id + "_environment"
        dace.library.environment(Environment)

        init, exit = cudnn_implementations._cudnn_tensor_descriptor_code(
            new_X_desc, f"{unique_id}_X_desc", False)
        Environment.init_code += init
        Environment.finalize_code += exit

        dX_desc = nsdfg.arrays[result.required_grad_names["X"]]
        init, exit = cudnn_implementations._cudnn_tensor_descriptor_code(
            dX_desc, f"{unique_id}_dX_desc", False)
        Environment.init_code += init
        Environment.finalize_code += exit

        dY_desc = nsdfg.arrays[result.given_grad_names["Y"]]
        init, exit = cudnn_implementations._cudnn_tensor_descriptor_code(
            dY_desc, f"{unique_id}_dY_desc", False)
        Environment.init_code += init
        Environment.finalize_code += exit

        # setup scale descriptor
        Environment.init_code += f"""
        __state->{unique_id}_dScale_desc = new cudnnTensorDescriptor_t; 
        daceml::cudnn::CheckCudnnError(cudnnCreateTensorDescriptor(__state->{unique_id}_dScale_desc));
        daceml::cudnn::CheckCudnnError(cudnnDeriveBNTensorDescriptor(
            *__state->{unique_id}_dScale_desc,
            *__state->{unique_id}_X_desc,
            CUDNN_BATCHNORM_SPATIAL));
        """
        Environment.finalize_code += f"""
        daceml::cudnn::CheckCudnnError(cudnnDestroyTensorDescriptor(*__state->{unique_id}_dScale_desc));
        delete __state->{unique_id}_dScale_desc;
        """

        tasklet_code = f"""
        {environments.cuDNN.handle_setup_code(forward_node)}
        float alpha = 1.f;
        float beta = 0.f;

        float _dYc;
        float _Xc;
        float _scalec;
        float _dXc;
        float _dScalec;
        float _dBiasc;

        cudaMemcpyAsync(&_dYc, _dY, 4, cudaMemcpyDeviceToHost, nullptr);
        cudaMemcpyAsync(&_Xc, _X, 4, cudaMemcpyDeviceToHost, nullptr);
        cudaMemcpyAsync(&_scalec, _scale, 4, cudaMemcpyDeviceToHost, nullptr);
        cudaMemcpyAsync(&_dXc, _dX, 4, cudaMemcpyDeviceToHost, nullptr);
        cudaMemcpyAsync(&_dScalec, _dScale, 4, cudaMemcpyDeviceToHost, nullptr);
        cudaMemcpyAsync(&_dBiasc, _dBias, 4, cudaMemcpyDeviceToHost, nullptr);

        daceml::cudnn::CheckCudnnError(cudnnBatchNormalizationBackward(
            __dace_cudnn_handle,
            CUDNN_BATCHNORM_SPATIAL,
            &alpha,
            &beta,
            &alpha,
            &beta,
            *__state->{unique_id}_X_desc,
            _X,
            *__state->{unique_id}_dY_desc,
            _dY,
            *__state->{unique_id}_dX_desc,
            _dX,
            *__state->{unique_id}_dScale_desc,
            _scale,
            _dScale,
            _dBias,
            {forward_node.epsilon},
            nullptr,
            nullptr));
        """

        in_connectors = ["X", "dY", "scale"]
        out_connectors = ["dX", "dScale", "dBias"]
        tasklet = nstate.add_tasklet(
            unique_id, {f"_{i}": dace.pointer(T)
                        for i in in_connectors},
            {f"_{i}": dace.pointer(T)
             for i in out_connectors}, tasklet_code, dtypes.Language.CPP)
        tasklet.environments = [Environment.__name__]

        # connect inputs
        arr_name = result.given_grad_names["Y"]
        nstate.add_edge(nstate.add_read(arr_name), None, tasklet, f"_dY",
                        nsdfg.make_array_memlet(arr_name))
        nstate.add_edge(nstate.add_read("X"), None, tasklet, f"_X",
                        nsdfg.make_array_memlet("X"))
        nstate.add_edge(nstate.add_read("scale"), None, tasklet, f"_scale",
                        nsdfg.make_array_memlet("scale"))

        # connect outputs
        arr_name = result.required_grad_names["X"]
        nstate.add_edge(tasklet, "_dX", nstate.add_write(arr_name), None,
                        nsdfg.make_array_memlet(arr_name))
        arr_name = result.required_grad_names["scale"]
        nstate.add_edge(tasklet, "_dScale", nstate.add_write(arr_name), None,
                        nsdfg.make_array_memlet(arr_name))
        arr_name = result.required_grad_names["B"]
        nstate.add_edge(tasklet, "_dBias", nstate.add_write(arr_name), None,
                        nsdfg.make_array_memlet(arr_name))

        node = context.backward_state.add_nested_sdfg(
            nsdfg, None, {"X", result.given_grad_names["Y"], "scale"},
            {result.required_grad_names[a]
             for a in {"X", "scale", "B"}})
        return node, result
