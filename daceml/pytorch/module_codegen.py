""" Code generation for PyTorch C++ dispatched operators. """
import copy
import dataclasses
import itertools
import operator
import os
from typing import List, Tuple, Callable, Optional, Dict, Union

import dace.library
import numpy as np
import torch
from dace import dtypes as dt, data
from dace.codegen import targets, compiler
from dace.codegen.codeobject import CodeObject
from dace.codegen.compiled_sdfg import CompiledSDFG
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.targets.common import sym2cpp

from daceml.autodiff import BackwardResult
from daceml.onnx.converters import clean_onnx_name
from daceml.onnx.onnx_importer import create_output_array
from daceml.pytorch.environments import PyTorch
from daceml.util import is_cuda, platform_library_name


@dataclasses.dataclass
class CompiledTorchFunction:
    """ A tuple holding the context for an executable function """
    function: Callable  #: the torch callable function
    compiled_sdfgs: List[
        CompiledSDFG]  #: the compiled SDFGs holding their states
    #: the pointers to the initialized SDFG state handles. Must be passed as the first arguments to function.
    ptr: List[torch.Tensor]


_REPLACED_CTYPES = {dace.int64: "int64_t", dace.uint64: "uint64_t"}


def torch_ctype(dtype: dace.typeclass) -> str:
    if dtype in _REPLACED_CTYPES:
        ctype = _REPLACED_CTYPES[dtype]
    else:
        ctype = dtype.ctype
    return ctype


def get_arglist(
        module: 'daceml.pytorch.DaceModule') -> Tuple[List[str], List[str]]:
    """ Get the list of forward-pass argument names for a module

        :param module: the module
        :return: the list of strings that are the argnames to the module, and the list of names of the outputs
    """
    arglist = [clean_onnx_name(i) for i in module.dace_model.inputs]

    # add any parameters that are required
    named_params = [
        clean_onnx_name(n) for n, _ in module.model.named_parameters()
    ]
    arglist.extend(n for n in named_params
                   if n not in arglist and n in module.sdfg.arrays
                   and not module.sdfg.arrays[n].transient)

    outputs = [clean_onnx_name(o) for o in module.dace_model.outputs]
    return arglist, outputs


_TYPECLASS_TO_TORCH_DTYPE_STR = {
    dt.int8: "kInt8",
    dt.uint8: "kUInt8",
    dt.int16: "kInt16",
    dt.int32: "kInt32",
    dt.int64: "kInt64",
    dt.float16: "kFloat16",
    dt.float32: "kFloat32",
    dt.float64: "kFloat64",
}


def tensor_init_for_desc(name: str, desc: data.Data, zeros=False) -> str:
    """ Emit the initialization code for a descriptor.
    """
    return f"""\
Tensor {name} = torch::{'zeros' if zeros else 'empty'}(
    {{{', '.join(str(s) for s in desc.shape)}}},
    torch::TensorOptions()
        .dtype(torch::{_TYPECLASS_TO_TORCH_DTYPE_STR[desc.dtype]})
        .device(torch::{'kCUDA' if is_cuda(desc.storage) else 'kCPU'})
        .layout(torch::kStrided));
    """


def initialize_outputs_code(module: 'daceml.pytorch.DaceModule',
                            output_names: List[str]) -> str:
    """ Generate the code that initializes the output tensors

        :param module: the module
        :param output_names: the output names of the SDFG.
        :param backward_arrays: names of array that must be saved for the backward pass. Only required if
                                generating code for a differentiable function.
        :return: the code
    """
    arglist = module.sdfg.arglist()
    code = ""
    for name in output_names:
        code += tensor_init_for_desc(name, arglist[name])

    return code


def argument_codegen(
        sdfg: dace.SDFG,
        clean_weights: Dict[str, torch.Tensor],
        input_names: List[str],
        output_names: List[str],
        guard_contiguous: Optional[List[str]] = None) -> Tuple[str, str, str]:
    """ Generate the code that grabs the pointers of inputs and outputs.

        The names of the tensors will match the SDFG tensor names. Tensors that are not created by us (i.e. inputs)
        should be named {sdfg_name}_ first, and then .contiguous() will be called on them to yield the tensor that we
        require. This is the case for all tensors in ``guard_contiguous``.

        :param module: the module
        :param clean_weights: the constant weights of the SDFG.
        :param input_names: names of inputs to the torch function.
        :param output_names: names of outputs to the torch function.
        :param guard_contiguous: a subset of input_names to call .contiguous on. If None, all input names will be
                                 guarded.
        :return: the code for initializing the argument, the sdfg arguments in order, and the init call arguments
    """
    arglist = sdfg.arglist()

    guard_contiguous = set(guard_contiguous or input_names)

    # initialize the inputs and outputs
    ptr_init_code = "\n// setup input and output pointers\n"
    for name in input_names:
        tctype = torch_ctype(arglist[name].dtype)
        dctype = arglist[name].dtype
        if isinstance(arglist[name], data.Array) or dt.can_access(
                dt.ScheduleType.GPU_Device, arglist[name].storage):
            if name in guard_contiguous:
                ptr_init_code += '\n' + f"Tensor {name} = {name}_.contiguous();"
            ptr_init_code += '\n' + f"{dctype} *{name}_ptr = reinterpret_cast<{dctype}*>({name}.data_ptr<{tctype}>());"

        elif isinstance(arglist[name], data.Scalar):
            if name in guard_contiguous:
                ptr_init_code += '\n' + f"{dctype} {name}_ptr = static_cast<{dctype}>({name}_.item().to<{tctype}>());"
            else:
                ptr_init_code += '\n' + f"{dctype} {name}_ptr = static_cast<{dctype}>({name}.item().to<{tctype}>());"
        else:
            raise ValueError(
                f"Unsupported data type {type(arglist[name])} for descriptor {name}"
            )

    ptr_init_code += '\n'

    # outputs and bwd arrays
    ptr_init_code += '\n'.join(
        f"{arglist[name].dtype.ctype} *{name}_ptr = reinterpret_cast<{arglist[name].dtype.ctype}*>"
        f"({name}.data_ptr<{torch_ctype(arglist[name].dtype)}>());"
        for name in output_names)
    ptr_init_code += "\n// setup constant arguments\n"

    # initialize all remaining parameters
    remaining = set(arglist).difference(
        itertools.chain(input_names, output_names))
    for name in remaining:
        # remaining args must be constants
        if name not in clean_weights:
            raise ValueError(
                f"Cannot generate PyTorch module C++ code: SDFG argument {name} is not an input or output"
                f" of the PyTorch Module, and not a constant.")
        if arglist[name].total_size > 1000:
            raise ValueError(
                f"Cannot generate PyTorch module C++ code: SDFG argument {name} is not an input or output"
                f" of the PyTorch Module, and is too large.")

        value = clean_weights[name]
        ptr_init_code += f"{constant_initializer_code(name, arglist[name], value)}\n"

    arguments = ", ".join(f"{n}_ptr" for n in arglist)
    init_arguments = ", ".join(f"{n}_ptr" for n, desc in arglist.items()
                               if isinstance(desc, data.Scalar))

    return ptr_init_code, arguments, init_arguments


def item_to_cpp_literal(item) -> str:
    dtype = str(item.dtype)
    if dtype == "float32":
        return f"{item}f"
    elif dtype == "int64":
        return f"{item}l"
    elif dtype in ["float64", "int32", "int16", "int8"]:
        return str(item)
    else:
        raise ValueError(f"Unsupported tensor type {item.dtype}")


def constant_initializer_code(name: str, desc: data.Data, value) -> str:
    gpu_storage = dt.can_access(dt.ScheduleType.GPU_Device, desc.storage)

    if isinstance(desc, data.Array) or gpu_storage:
        iterator = np.nditer(value.cpu().numpy(), order="C")
        gpu_copy_code = f"""
        Tensor {name} = torch::from_blob({name}_ptr_cpu, {{{', '.join(sym2cpp(s) for s in desc.shape)}}},
            {{{', '.join(sym2cpp(s) for s in desc.strides)}}}, torch::{_TYPECLASS_TO_TORCH_DTYPE_STR[desc.dtype]})
            .to(torch::kCUDA);
        {desc.dtype.ctype} *{name}_ptr = reinterpret_cast<{desc.dtype.ctype}*>({name}.data_ptr<{torch_ctype(desc.dtype)}>());
        """
        return f"""
        {desc.dtype.ctype} {name}_ptr{'_cpu' if gpu_storage else ''}[{sym2cpp(desc.total_size)}] =
            {{{', '.join(item_to_cpp_literal(e) for e in iterator)}}};
        {gpu_copy_code if gpu_storage else ""}
        """
    elif isinstance(desc, data.Scalar):
        return f"{desc.dtype.ctype} {name}_ptr = {str(value.item())};"
    else:
        raise ValueError("Unsupported data descriptor")


def return_type_str(outputs: List[str]) -> str:
    return f"""{"Tensor" if len(outputs) == 1 else f"std::tuple<{', '.join(['Tensor'] * len(outputs))}>"}"""


def save_non_inputs_outputs(names: List[str]):
    return "\n".join(f'ctx->saved_data["{n}"] = {n};' for n in names)


def recover_saved_inputs_outputs(saved_inputs_outputs: List[str],
                                 other_saved: List[str]):
    code = ""
    if saved_inputs_outputs:
        code += "auto saved = ctx->get_saved_variables();\n"
        for i, n in enumerate(saved_inputs_outputs):
            code += f"\nauto {n} = saved[{i}];"

    for n in other_saved:
        code += f'\nauto {n} = ctx->saved_data["{n}"].toTensor();'

    return code


def setup_grad_values(backward_result: BackwardResult, sdfg: dace.SDFG,
                      outputs: List[str]) -> str:
    code = "// input grads"
    for _, grad_name in backward_result.required_grad_names.items():
        code += "\n" + tensor_init_for_desc(
            grad_name, sdfg.arrays[grad_name], zeros=True)

    code += "// output grads"
    for i, o in enumerate(outputs):
        grad_name = backward_result.given_grad_names[o]
        code += f'\nauto {grad_name}_ = grad_outputs[{i}];'

    return code


def code_for_backward_function(module: 'daceml.pytorch.DaceModule',
                               forward_sdfg: dace.SDFG,
                               backward_sdfg: dace.SDFG,
                               backward_result: BackwardResult,
                               forwarded_arrays: Dict[str, data.Data]) -> str:

    inputs, outputs = get_arglist(module)
    sdfg_name = forward_sdfg.name

    ret_str = return_type_str(outputs)

    outputs_with_forwarded_outputs = copy.deepcopy(outputs)
    outputs_with_forwarded_outputs.extend(
        n for n in forwarded_arrays if n not in inputs and n not in outputs)

    fwd_ptr_init_code, fwd_sdfg_call_arguments, _ = argument_codegen(
        forward_sdfg, module.dace_model.clean_weights, inputs,
        outputs_with_forwarded_outputs)

    # inputs are given_grads + forwarded_outputs
    bwd_inputs = list(
        backward_result.given_grad_names.values()) + list(forwarded_arrays)

    # outputs are required grads
    bwd_outputs = list(backward_result.required_grad_names.values())

    bwd_ptr_init_code, bwd_sdfg_call_arguments, _ = argument_codegen(
        backward_sdfg,
        module.dace_model.clean_weights,
        bwd_inputs,
        bwd_outputs,
        guard_contiguous=list(backward_result.given_grad_names.values()))

    # saved inputs/outputs
    saved_io_for_backward = [
        n for n in forwarded_arrays if n in inputs or n in outputs
    ]
    other_saved_for_backward = [
        n for n in forwarded_arrays if n not in inputs and n not in outputs
    ]
    return f"""
{get_header(forward_sdfg, backward_sdfg, inputs, outputs, module.use_cuda)}
class {sdfg_name}Function : public torch::autograd::Function<{sdfg_name}Function> {{
    public:
        static
            {ret_str}
            forward(
            AutogradContext *ctx,
            int64_t fwd_handle_ptr, int64_t bwd_handle_ptr, {", ".join(f"const Tensor& {name}_" for name in inputs)}) {{

            at::AutoNonVariableTypeMode g;

            // initialize outputs
            {initialize_outputs_code(module, outputs_with_forwarded_outputs)}

            {fwd_ptr_init_code}

            // get SDFG state handle
            {forward_sdfg.name}Handle_t handle = reinterpret_cast<{forward_sdfg.name}Handle_t>(fwd_handle_ptr);


            // call SDFG
            __program_{forward_sdfg.name}(handle, {fwd_sdfg_call_arguments});

            // save inputs/outputs for backward
            {
                f"ctx->save_for_backward({{{', '.join(f'{n}' for n in saved_io_for_backward)}}});" 
                if saved_io_for_backward else ""
            }

            // save non-inputs/outputs
            {save_non_inputs_outputs(other_saved_for_backward)}

            // save bwd handle
            ctx->saved_data["bwd_handle"] = bwd_handle_ptr;

            // return to torch
            return {f"{outputs[0]}" if len(outputs) == 1
            else f"{{{', '.join(o for o in outputs)}}}"};
        }}
        
        static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {{
            // recover bwd_handle_ptr
            int64_t bwd_handle_ptr = ctx->saved_data.find("bwd_handle")->second.toInt();

            // recover saved values
            {recover_saved_inputs_outputs(saved_io_for_backward, other_saved_for_backward)}

            // create grad values
            // NOTE, it might make sense take these from .grad()
            {setup_grad_values(backward_result, backward_sdfg, outputs)}
            
            {bwd_ptr_init_code}

            // get SDFG state handle
            {backward_sdfg.name}Handle_t handle = reinterpret_cast<{backward_sdfg.name}Handle_t>(bwd_handle_ptr);

            // call bwd SDFG
            __program_{backward_sdfg.name}(handle, {bwd_sdfg_call_arguments});
            
            // return calculated grads in correct order
            // first two grads are None (these are the grads for the handle ptrs)
            return {{
                Tensor(), Tensor(), {', '.join(backward_result.required_grad_names[i] for i in inputs)}
            }};
        }}
}};

{ret_str}
{sdfg_name}_autograd(int64_t handle_ptr, int64_t bwd_handle_ptr, {",".join(f"const Tensor& {name}_" for name in inputs)}) {{
    return {sdfg_name}Function::apply(
        handle_ptr, bwd_handle_ptr, {", ".join(f"{name}_" for name in inputs)}
    );
}}

TORCH_LIBRARY_IMPL(daceml_{sdfg_name}, Autograd{'CUDA' if module.use_cuda else 'CPU'}, m) {{
    m.impl("{sdfg_name}", {sdfg_name}_autograd);
}}
        """


def code_for_module(module: 'daceml.pytorch.DaceModule',
                    compiled_sdfg: CompiledSDFG) -> str:
    """ Generate the code for an operator that calls the sdfgs in the module.

        :param module: the module.
        :param compiled_sdfg: the compiled SDFG.
    """

    inputs, outputs = get_arglist(module)
    sdfg_name = compiled_sdfg.sdfg.name

    ret_str = return_type_str(outputs)
    if module.backward:
        raise NotImplemented("todo")
    else:
        ptr_init_code, sdfg_call_arguments, init_arguments = argument_codegen(
            compiled_sdfg.sdfg, module.dace_model.clean_weights, inputs,
            outputs)
        return f"""
{get_header(compiled_sdfg.sdfg, None, inputs, outputs, module.use_cuda)}

// function definition
{ret_str}
{sdfg_name}(int64_t handle_ptr, {",".join(f"const Tensor& {name}_" for name in inputs)}) {{

    // initialize outputs
    {initialize_outputs_code(module, outputs)}
    
    {ptr_init_code}

    // get SDFG state handle
    {sdfg_name}Handle_t handle = reinterpret_cast<{sdfg_name}Handle_t>(handle_ptr);

    // call SDFG
    __program_{sdfg_name}(handle, {sdfg_call_arguments});

    // return to torch
    return {f"{outputs[0]}" if len(outputs) == 1
        else f"{{{', '.join(o for o in outputs)}}}"};
}}

TORCH_LIBRARY_IMPL(daceml_{sdfg_name}, {'CUDA' if module.use_cuda else 'CPU'}, m) {{
    m.impl("{sdfg_name}", {sdfg_name});
}}
        """


def get_header(fwd_sdfg: dace.SDFG, bwd_sdfg: Optional[dace.SDFG], inputs,
               outputs, use_cuda: bool) -> str:
    return f"""
#include <torch/torch.h>
#include <torch/script.h>
#include "{fwd_sdfg.name}.h"
{"" if bwd_sdfg is None else f'#include "{bwd_sdfg.name}.h"'}
using torch::Tensor;
using torch::DeviceType;
using torch::autograd::tensor_list;
using torch::autograd::AutogradContext;

TORCH_LIBRARY(daceml_{fwd_sdfg.name}, m) {{
    m.def("{fwd_sdfg.name}(int handle_ptr,{"int bwd_handle_ptr," if bwd_sdfg else ""} {", ".join('Tensor ' + arg for arg in inputs)}) -> {'Tensor' if len(outputs) == 1
    else "(" + ", ".join(['Tensor'] * len(outputs)) + ")"}");
}}
"""


def compile_and_get_function(module: 'daceml.pytorch.DaceModule',
                             dummy_inputs) -> CompiledTorchFunction:
    """ Get a torch callable for the module. This will compile the sdfg, compile a PyTorch C++ operator, register it
        with PyTorch and return the function that calls it.

        This function handles code generation for both the forward and backward pass.

        :param module: the module.
        :param dummy_inputs: dummy inputs to initialize the model with.
        :return: the callable function for the SDFG.
    """

    # build the SDFG
    # set all states to not-sync
    for state in module.sdfg.nodes():
        state.nosync = True

    environments = {
        PyTorch.full_class_path(),
    }
    if module.backward:
        compiled, handle_ptr, compiled_bwd, bwd_handle_ptr = compile_and_init_sdfgs(
            module, dummy_inputs)

        compiled_sdfgs = [compiled, compiled_bwd]
        ptrs = [handle_ptr, bwd_handle_ptr]
        environments.add(get_env_for_sdfg(compiled_bwd).full_class_path())
        code = code_for_backward_function(module, compiled.sdfg,
                                          compiled_bwd.sdfg, module._ad_result,
                                          module._ad_inp_arrs)
    else:
        compiled, handle_ptr = compile_and_init_sdfgs(module, dummy_inputs)
        ptrs = [handle_ptr]
        code = code_for_module(module, compiled)
        compiled_sdfgs = [compiled]
    environments.add(get_env_for_sdfg(compiled).full_class_path())
    code = indent_code(code)

    # build the PyTorch module
    libname = f"torch_{module.sdfg.name}"
    program = CodeObject(libname,
                         code,
                         "cpp",
                         targets.cpu.CPUCodeGen,
                         f"Torch{module.sdfg_name}",
                         environments=environments)
    torch_module_build_path = os.path.join('.dacecache',
                                           f"torch_{compiled.sdfg.name}")

    compiler.generate_program_folder(None, [program], torch_module_build_path)
    compiler.configure_and_compile(torch_module_build_path)

    torch.ops.load_library(
        os.path.join(torch_module_build_path, "build",
                     platform_library_name(libname)))

    torch_function = operator.attrgetter(
        f"daceml_{module.sdfg.name}.{module.sdfg.name}")(torch.ops)

    result = CompiledTorchFunction(function=torch_function,
                                   compiled_sdfgs=compiled_sdfgs,
                                   ptr=ptrs)
    return result


def compile_and_init_sdfgs(
    module: 'daceml.pytorch.DaceModule', dummy_inputs
) -> (Union[Tuple[CompiledSDFG, int], Tuple[CompiledSDFG, int, CompiledSDFG,
                                            int]]):

    compiled: CompiledSDFG = module.dace_model.compile_and_init()
    # construct the arguments and initialize the SDFG
    args = tuple(dummy_inputs) + tuple(
        p.data for n, p in module.model.named_parameters()
        if n in module.dace_model.inputs)
    inputs, symbols, outputs = module.dace_model._call_args(args=args,
                                                            kwargs={})

    if module.backward:
        forwarded_transients = {
            name: create_output_array(symbols,
                                      desc,
                                      use_torch=True,
                                      zeros=True)
            for name, desc in module._ad_inp_arrs.items()
        }
    else:
        forwarded_transients = {}

    _, initargtuple = compiled._construct_args({
        **inputs,
        **outputs,
        **symbols,
        **forwarded_transients,
        **module.dace_model.initialized_parameters
    })
    compiled.initialize(*initargtuple)
    for _, hook in module.post_compile_hooks.items():
        hook(compiled)
    handle_ptr = torch.tensor([compiled._libhandle.value]).squeeze(0)

    if module.backward:
        # compile and initialize the backward_sdfg
        compiled_bwd: CompiledSDFG = module.backward_sdfg.compile()

        required_grads = {
            bwd_name: create_output_array(symbols,
                                          compiled_bwd.sdfg.arrays[bwd_name],
                                          use_torch=True,
                                          zeros=True)
            for _, bwd_name in module._ad_result.required_grad_names.items()
        }
        given_grads = {
            bwd_name: create_output_array(symbols,
                                          compiled_bwd.sdfg.arrays[bwd_name],
                                          use_torch=True,
                                          zeros=True)
            for _, bwd_name in module._ad_result.given_grad_names.items()
        }

        _, initargtuple = compiled_bwd._construct_args({
            **required_grads,
            **given_grads,
            **forwarded_transients
        })
        compiled_bwd.initialize(*initargtuple)
        bwd_handle_ptr = torch.tensor([compiled_bwd._libhandle.value
                                       ]).squeeze(0)
        return compiled, handle_ptr, compiled_bwd, bwd_handle_ptr
    else:
        return compiled, handle_ptr


def get_env_for_sdfg(compiled: CompiledSDFG):

    sdfg_build_path = os.path.abspath(compiled.sdfg.build_folder)

    class SDFGEnvironment:
        """ Environment for the SDFG
        """

        cmake_minimum_version = None
        cmake_packages = []
        cmake_variables = {}
        cmake_includes = [os.path.join(sdfg_build_path, "include")]
        cmake_compile_flags = []
        cmake_link_flags = []
        cmake_files = []
        cmake_libraries = [
            os.path.join(sdfg_build_path, "build",
                         platform_library_name(compiled.sdfg.name))
        ]
        state_fields = []
        dependencies = []
        headers = []
        init_code = ""
        finalize_code = ""

    SDFGEnvironment.__name__ = compiled.sdfg.name
    dace.library.environment(SDFGEnvironment)
    return SDFGEnvironment


def indent_code(code: str) -> str:
    stream = CodeIOStream()
    stream.write(code)
    return stream.getvalue()
