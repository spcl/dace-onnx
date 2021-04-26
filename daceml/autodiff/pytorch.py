import logging
from collections import OrderedDict
from typing import Type, Tuple, Dict, Callable

import dace
import torch
from dace import data as dt
from dace.codegen import compiled_sdfg

from daceml.autodiff.backward_pass_generator import BackwardPassGenerator
from daceml.autodiff.base_abc import AutoDiffException
from daceml.onnx.converters import clean_onnx_name
from daceml.onnx.onnx_importer import create_output_array
from daceml.pytorch.module import DaceModule

log = logging.getLogger(__name__)

hook_type = Dict[str, Callable[
    [compiled_sdfg.CompiledSDFG, compiled_sdfg.CompiledSDFG], None]]


def make_backward_function(
        model: DaceModule,
        apply_strict=False) -> Tuple[hook_type, Type[torch.autograd.Function]]:
    """ Convert an ONNXModel to a PyTorch differentiable function. This method should not be used on it's own.
        Instead use the ``backward=True`` parameter of :class:`daceml.pytorch.DaceModule`.

        :param model: the model to convert.
        :param apply_strict: whether to apply strict transformations before creating the backward pass.
        :return: the PyTorch compatible :class:`torch.autograd.Function`.
    """

    if len(model.sdfg.nodes()) != 1:
        raise AutoDiffException(
            "Expected to find exactly one SDFGState, found {}".format(
                len(model.sdfg.nodes())))

    forward_sdfg = model.sdfg
    forward_state = model.sdfg.nodes()[0]

    backward_sdfg = dace.SDFG(forward_sdfg.name + "_backward")
    backward_state = backward_sdfg.add_state()

    # we want gradients for all inputs that are not pytorch buffers
    named_buffers = {n for n, _ in model.pytorch_model.named_buffers()}
    required_gradients = [
        clean_onnx_name(name) for name in model.dace_onnx_model.inputs
        if name not in named_buffers
    ]
    named_parameters = dict(model.pytorch_model.named_parameters())
    required_gradients.extend(
        clean_onnx_name(name) for name, param in named_parameters.items()
        if param.requires_grad)
    required_gradients = list(set(required_gradients))

    gen = BackwardPassGenerator(sdfg=forward_sdfg,
                                state=forward_state,
                                given_gradients=[
                                    clean_onnx_name(name)
                                    for name in model.dace_onnx_model.outputs
                                ],
                                required_gradients=required_gradients,
                                backward_sdfg=backward_sdfg,
                                backward_state=backward_state,
                                apply_strict=apply_strict)

    backward_result, backward_grad_arrays, backward_input_arrays = gen.backward(
    )

    replaced_scalars = {}
    for name, desc in backward_input_arrays.items():
        if name in named_parameters:
            # these may be non-transient, and will be passed separately if required
            continue

        if name not in forward_sdfg.arrays:
            raise AutoDiffException(
                "Expected to find array with name '{}' in SDFG".format(name))

        forward_desc = forward_sdfg.arrays[name]
        # we will save this output and pass it to the backward pass

        # Views should not be forwarded. Instead the backward pass generator should forward the source of the view,
        # and rebuild the sequence of required views in the backward pass.
        assert type(forward_desc) is not dt.View
        if isinstance(forward_desc, dt.Scalar):
            # we can't return scalars from SDFGs, so we add a copy to an array of size 1
            fwd_arr_name, _ = forward_sdfg.add_array(
                name + "_array", [1],
                forward_desc.dtype,
                transient=False,
                storage=forward_desc.storage,
                find_new_name=True)
            bwd_arr_name, _ = backward_sdfg.add_array(
                name + "_array", [1],
                forward_desc.dtype,
                transient=False,
                storage=forward_desc.storage,
                find_new_name=True)
            backward_sdfg.arrays[name].transient = True

            fwd_copy_state = forward_sdfg.add_state_after(forward_state,
                                                          label="copy_out_" +
                                                          fwd_arr_name)
            bwd_copy_state = backward_sdfg.add_state_before(backward_state,
                                                            label="copy_in_" +
                                                            bwd_arr_name)
            fwd_copy_state.add_edge(fwd_copy_state.add_read(name), None,
                                    fwd_copy_state.add_write(fwd_arr_name),
                                    None, dace.Memlet(name + "[0]"))

            bwd_copy_state.add_edge(bwd_copy_state.add_read(bwd_arr_name),
                                    None, bwd_copy_state.add_write(name), None,
                                    dace.Memlet(name + "[0]"))
            replaced_scalars[name] = fwd_arr_name
        else:
            forward_sdfg.arrays[name].transient = False

    backward_sdfg.view()
    backward_sdfg.validate()

    # initialization of the SDFGs
    forward_sdfg.validate()

    # these are the grads we will calculate
    input_grad_names = [
        backward_result.required_grad_names[clean_onnx_name(inp)]
        for inp in required_gradients if inp not in named_parameters
    ]

    required_parameter_gradients = []
    for inp in required_gradients:
        if backward_result.required_grad_names[inp] not in input_grad_names:
            dace_name, pt_name, param = backward_result.required_grad_names[
                clean_onnx_name(inp)], inp, named_parameters[inp]
            if param.grad is None:
                param.grad = create_output_array(
                    {},
                    backward_sdfg.arrays[dace_name],
                    use_torch=True,
                    zeros=True)
            required_parameter_gradients.append((dace_name, pt_name, param))

    class DaceFunction(torch.autograd.Function):
        _forward_model = model.dace_onnx_model
        _backward_result = backward_result
        _forward_sdfg = forward_sdfg
        _backward_sdfg = backward_sdfg

        @staticmethod
        def forward(ctx, *args):
            # setup the intermediate buffers

            # if any(not inp.is_contiguous() for inp in args):
            #     log.warning("forced to copy input since it was not contiguous")

            copied_args = tuple(inp if inp.is_contiguous else inp.contiguous()
                                for inp in args)

            # prepare the arguments
            inputs, params, symbols, outputs = DaceFunction._forward_model._call_args(
                args=copied_args, kwargs={})

            # create the empty tensors we need for the intermediate values
            for inp, val in backward_input_arrays.items():
                if isinstance(val, dt.Scalar):
                    # the value we need is actually in an array
                    inp = replaced_scalars[inp]

                if inp not in inputs and inp not in outputs and inp not in params:
                    inputs[inp] = create_output_array(symbols,
                                                      forward_sdfg.arrays[inp],
                                                      use_torch=True)

            DaceFunction._forward_sdfg(**inputs, **symbols, **params,
                                       **outputs)

            def _get_arr(name, desc):

                if isinstance(desc, dt.Scalar):
                    name = replaced_scalars[name]

                if name in named_parameters:
                    value = named_parameters[name]
                elif name in inputs:
                    value = inputs[name]
                elif name in outputs:
                    value = outputs[name]
                elif name in params:
                    value = params[name]
                else:
                    raise AutoDiffException(
                        f"Could not get value of array {name}")

                return value

            # save the arrays we need for the backward pass
            backward_inputs = {
                name: _get_arr(name, desc)
                for name, desc in backward_input_arrays.items()
            }
            for name in replaced_scalars:
                backward_inputs[replaced_scalars[name]] = backward_inputs[name]
                del backward_inputs[name]
            ctx.dace_backward_inputs = backward_inputs
            ctx.dace_symbols = symbols

            if len(outputs) == 1:
                return next(iter(outputs.values()))

            return tuple(outputs.values())

        @staticmethod
        def backward(ctx, *grads):
            backward_inputs = ctx.dace_backward_inputs

            if len(grads) != len(model.dace_onnx_model.outputs):
                raise ValueError("Expected to receive {} grads, got {}".format(
                    len(model.dace_onnx_model.outputs), len(grads)))

            given_grads = dict(
                zip((DaceFunction._backward_result.given_grad_names[
                    clean_onnx_name(outp)]
                     for outp in model.dace_onnx_model.outputs), grads))
            for name, value in given_grads.items():
                if not isinstance(value, torch.Tensor):
                    raise ValueError(
                        "Unsupported input with type {};"
                        " currently only tensor inputs are supported".format(
                            type(value)))
                if not value.is_contiguous():
                    log.warning(
                        "forced to copy input since it was not contiguous")
                    given_grads[name] = value.contiguous()

            parameter_grad_values = {
                dace_name: param.grad
                for dace_name, pt_name, param in required_parameter_gradients
            }

            # init the grads we will calculate with zeros
            input_grad_values = {}
            for name in input_grad_names:
                input_grad_values[clean_onnx_name(name)] = create_output_array(
                    ctx.dace_symbols,
                    backward_grad_arrays[name],
                    use_torch=True,
                    zeros=True)

            DaceFunction._backward_sdfg(**input_grad_values,
                                        **parameter_grad_values,
                                        **backward_inputs, **given_grads)

            return tuple(input_grad_values.values())

    return gen.post_compile_hooks, DaceFunction
