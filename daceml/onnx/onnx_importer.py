import collections
import logging
from itertools import chain, repeat
from typing import Dict, Union, Tuple, Any, List, Optional, OrderedDict, Callable

import numpy as np
import torch

import onnx
from dace.codegen import compiled_sdfg
from onnx import numpy_helper

import dace
from dace import data as dt, dtypes, nodes, SDFG, SDFGState
from dace.frontend.python.parser import infer_symbols_from_shapes
from dace.symbolic import pystr_to_symbolic
from dace.transformation import dataflow

from daceml import transformation
from daceml.onnx.shape_inference import shape_inference
from daceml.onnx.converters import convert_attribute_proto, onnx_tensor_type_to_typeclass, clean_onnx_name
from daceml.onnx.schema import ONNXParameterType
from daceml.onnx.nodes.onnx_op import get_onnx_node, has_onnx_node, ONNXOp
from daceml.util import utils, is_cuda

log = logging.getLogger(__name__)

numpy_to_torch_dtype_dict = {
    np.bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128
}

torch_to_numpy_dtype_dict = {
    v: k
    for k, v in numpy_to_torch_dtype_dict.items()
}


def _nested_HasField(obj, full_attr):
    """Performs a nested hasattr check, separating attr on dots."""
    attrs = full_attr.split(".")
    for attr in attrs:
        if obj.HasField(attr):
            obj = getattr(obj, attr)
        else:
            return False
    return True


class ONNXModel:
    """ Loads an ONNX model into an SDFG.

        :Example:
            First download an ONNX model, such as
            `efficientnet <http://spclstorage.inf.ethz.ch/~rauscho/efficientnet-lite4-11.onnx>`_.

            .. testsetup::

                import subprocess
                model_path = os.path.join("..", "tests", "onnx_files", "efficientnet.onnx")
                # Download model
                if not os.path.exists(model_path):
                    subprocess.check_call([
                        "wget",
                        "http://spclstorage.inf.ethz.ch/~rauscho/efficientnet-lite4-11.onnx",
                        "--output-document={}".format(model_path),
                        "--no-verbose"
                    ])


            .. testcode::

                import onnx
                import os
                import numpy as np
                from daceml.onnx import ONNXModel

                model_path = os.path.join("..", "tests", "onnx_files", "efficientnet.onnx")
                model = onnx.load(model_path)
                dace_model = ONNXModel("efficientnet", model)

                test_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
                dace_model(test_input)

    """
    def __init__(self,
                 name: str,
                 model: onnx.ModelProto,
                 infer_shapes: bool = True,
                 cuda: bool = False,
                 apply_strict: bool = False,
                 auto_optimize: bool = True,
                 fold_constants: bool = True,
                 parent_pytorch_module: Optional[torch.nn.Module] = None,
                 storage: Optional[dtypes.StorageType] = None,
                 save_transients: Optional[Dict[str, torch.Tensor]] = None,
                 auto_merge: bool = False):
        """
        :param name: the name for the SDFG.
        :param model: the model to import.
        :param infer_shapes: whether to infer shapes for the model. If this is ``False``, the model must have
                             value infos (with shapes) for all arrays, including intermediate values.
        :param cuda: if ``True``, the model will be executed on the GPU.
        :param apply_strict: if ``True``, apply strict transformations after all nodes have
                             been expanded calling (warning: this can be very slow!)
        :param auto_optimize: if ``True``, apply automatic optimizations before calling.
        :param parent_pytorch_module: when not None, the weight tensors are loaded from the parameters of this model
                                      rather than the ONNX graph.
        :param storage: the storage type of the parameters, inputs and outputs. If None, will be set according to
                        ``cuda``.
        :param save_transients: if not None, save transients to this dict (for debugging).
        :param auto_merge: whether to automatically merge symbolic shapes in symbolic shape inference.
        """

        for opset in model.opset_import:
            if opset.domain == "" and opset.version != 12:
                log.warning(
                    f"Expected the onnx model to be exported with opset 12, got {opset.version}. This model may fail "
                    f"to import as a result.")

        self.do_auto_optimize = auto_optimize
        if infer_shapes:
            model = shape_inference.infer_shapes(model, auto_merge=auto_merge)

        graph: onnx.GraphProto = model.graph
        self.save_transients = save_transients
        self.sdfg: SDFG = SDFG(name)  #: the generated SDFG.
        self.sdfg._parent_onnx_model = self
        self.cuda = cuda
        self.apply_strict = apply_strict
        self.fold_constants = fold_constants
        self.state: SDFGState = self.sdfg.add_state(
        )  #: the state containing the model computation.

        # Add all values to the SDFG, check for unsupported ops
        ##########################################

        self.value_infos = {}

        self.inputs: List[str] = []  #: the inputs to the model
        self.outputs: List[str] = []  #: the outputs of the model

        if storage is None:
            storage = dtypes.StorageType.GPU_Global if self.cuda else dtypes.StorageType.Default

        for value, is_input in chain(zip(graph.input, repeat(True)),
                                     zip(graph.output, repeat(False))):
            if not value.HasField("name"):
                raise ValueError("Got input or output without name")
            if is_input:
                self.inputs.append(value.name)
            else:
                self.outputs.append(value.name)

            self.value_infos[value.name] = value
            storage = storage
            self._add_value_info(value, storage=storage)

        for value in graph.value_info:
            if not value.HasField("name"):
                raise ValueError("Got input or output without name")
            if value.name not in self.value_infos:
                self.value_infos[value.name] = value

        # add weights
        self.weights: Dict[str, torch.Tensor] = {
        }  #: mapping from weight name to array
        for init in graph.initializer:
            self._add_constant_tensor(init, parent_pytorch_module, storage)

        access_nodes = {}
        self._idx_to_node = []
        for i, node in enumerate(graph.node):
            if not has_onnx_node(node.op_type):
                raise ValueError("Unsupported ONNX operator: '{}'".format(
                    node.op_type))

            # extract the op attributes
            op_attributes = {
                attribute_proto.name: convert_attribute_proto(attribute_proto)
                for attribute_proto in node.attribute
            }

            if node.op_type == "Constant":
                # Add constants to weights immediately
                possible_values = [
                    "sparse_value", "value", "value_float", "value_floats",
                    "value_int", "value_ints", "value_string", "value_strings"
                ]

                # do some manual validation here since the node validation will never run
                if set(op_attributes).difference(possible_values):
                    raise ValueError(
                        f"Got unexpected attributes on Constant node "
                        f"{set(op_attributes).difference(possible_values)}")

                if len(op_attributes) != 1:
                    raise ValueError(
                        "Expected Constant node to have exactly one of its attributes set"
                    )

                if len(node.input) != 0 or len(node.output) != 1:
                    raise ValueError(
                        "Expected Constant node to have no inputs and exactly 1 output"
                    )

                value_name = next(iter(op_attributes))

                if node.output[0] not in self.value_infos:
                    raise ValueError(
                        "Could not find array with name '{}'".format(
                            node.output[0]))
                self._add_value_info(self.value_infos[node.output[0]],
                                     storage=storage)
                self.sdfg.arrays[clean_onnx_name(
                    node.output[0])].transient = False

                self.weights[node.output[0]] = torch.from_numpy(
                    op_attributes[value_name].copy())
                continue

            if node.HasField("name"):
                node_name = clean_onnx_name(node.name)
            else:
                node_name = node.op_type + "_" + str(i)

            # construct the dace node
            op_node = get_onnx_node(node.op_type)(node_name, **op_attributes)
            self.state.add_node(op_node)
            self._idx_to_node.append(op_node)

            for param_idx, (name, is_input) in chain(
                    enumerate(zip(node.input, repeat(True))),
                    enumerate(zip(node.output, repeat(False)))):
                if clean_onnx_name(name) not in self.sdfg.arrays:
                    if name not in self.value_infos:
                        raise ValueError(
                            "Could not find array with name '{}'".format(name))
                    self._add_value_info(self.value_infos[name])
                # get the access node
                if name in access_nodes:
                    access = access_nodes[name]
                    self._update_access_type(access, is_input)
                else:
                    access = nodes.AccessNode(
                        clean_onnx_name(name), dtypes.AccessType.ReadOnly
                        if is_input else dtypes.AccessType.WriteOnly)
                    self.state.add_node(access)
                    access_nodes[name] = access

                # get the connector name
                params = op_node.schema.inputs if is_input else op_node.schema.outputs
                params_len = len(params)
                if param_idx >= params_len:
                    # this is a variadic parameter. Then the last parameter of the parameter must be variadic.
                    if params[-1].param_type != ONNXParameterType.Variadic:
                        raise ValueError(
                            "Expected the last {i_or_o} parameter to be variadic,"
                            " since the {i_or_o} with idx {param_idx} has more parameters than the schema ({params_len})"
                            .format(i_or_o="input" if is_input else "output",
                                    param_idx=param_idx,
                                    params_len=params_len))
                    conn_name = params[-1].name + "__" + str(param_idx -
                                                             params_len + 1)
                elif params[
                        param_idx].param_type == ONNXParameterType.Variadic:
                    # this is a variadic parameter, and it is within the range of params, so it must be the first
                    # instance of a variadic parameter
                    conn_name = params[param_idx].name + "__0"
                else:
                    conn_name = params[param_idx].name

                data_desc = self.sdfg.arrays[clean_onnx_name(name)]

                # add the connector if required, and add an edge
                if is_input:
                    if conn_name not in op_node.in_connectors:
                        assert op_node.add_in_connector(conn_name)
                    self.state.add_edge(
                        access, None, op_node, conn_name,
                        dace.Memlet.from_array(clean_onnx_name(name),
                                               data_desc))
                else:
                    if conn_name not in op_node.out_connectors:
                        assert op_node.add_out_connector(conn_name)

                    self.state.add_edge(
                        op_node, conn_name, access, None,
                        dace.Memlet.from_array(clean_onnx_name(name),
                                               data_desc))

        if self.fold_constants:
            log.debug("Applying constant folding")
            self.sdfg.apply_transformations_repeated([
                transformation.ConstantFolding, dataflow.RedundantSecondArray
            ],
                                                     validate_all=True,
                                                     strict=True)

        if self.cuda:
            self.sdfg.apply_gpu_transformations()

    @staticmethod
    def _update_access_type(node: dace.nodes.AccessNode, is_input: bool):
        if node.access == dtypes.AccessType.ReadOnly and not is_input:
            node.access = dtypes.AccessType.ReadWrite
        elif node.access == dtypes.AccessType.WriteOnly and is_input:
            node.access = dtypes.AccessType.ReadWrite

    def _add_constant_tensor(self, tensor: onnx.TensorProto, parent_pt_model,
                             storage: dtypes.StorageType):
        if not tensor.HasField("name"):
            raise ValueError("Got tensor without name")

        if not tensor.HasField("data_type"):
            raise ValueError("Initializer tensor '{}' has no type".format(
                tensor.name))

        if tensor.name in self.inputs:
            # do not duplicate a weight if it is already an input
            return

        name = clean_onnx_name(tensor.name)

        dtype = onnx_tensor_type_to_typeclass(tensor.data_type)

        if len(tensor.dims) == 0:
            # this is a scalar
            self.sdfg.add_scalar(name, dtype, storage=storage)
        else:
            dims = [d for d in tensor.dims]
            if name not in self.sdfg.arrays:
                self.sdfg.add_array(name, dims, dtype, storage=storage)
            else:
                existing_arr = self.sdfg.arrays[name]
                if existing_arr.dtype != dtype:
                    raise ValueError(
                        "Invalid ONNX model; found two values with name '{}', but different dtypes ({} and {})"
                        .format(name, existing_arr.dtype, dtype))
                if tuple(existing_arr.shape) != tuple(dims):
                    raise ValueError(
                        "Invalid ONNX model; found two values with name '{}', but different dimensions ({} and {})"
                        .format(name, existing_arr.shape, dims))

        weight_arr = numpy_helper.to_array(tensor)

        if parent_pt_model is not None:
            parent_parameters = dict(parent_pt_model.named_parameters())

        if parent_pt_model is not None and tensor.name in parent_parameters:
            self.weights[tensor.name] = parent_parameters[tensor.name].data
        else:
            # we need to copy here because the weight_arr tensor is not writable
            self.weights[tensor.name] = torch.from_numpy(weight_arr.copy())

    def _add_value_info(self,
                        value_info: onnx.ValueInfoProto,
                        storage=dtypes.StorageType.Default):
        if not value_info.HasField("name"):
            raise ValueError("Got value without name")

        name = value_info.name

        if not _nested_HasField(value_info, "type.tensor_type.shape"):
            raise ValueError(
                "Value '{}' does not have a shape in this graph."
                " Please run shape inference before importing.".format(name))

        tensor_type = value_info.type.tensor_type

        if not tensor_type.HasField("elem_type"):
            raise ValueError(
                "Value '{}' does not have a type in this graph."
                " Please run type inference before importing.".format(name))

        shape = []
        for d in tensor_type.shape.dim:
            if d.HasField("dim_value"):
                shape.append(d.dim_value)
            elif d.HasField("dim_param"):
                parsed = pystr_to_symbolic(d.dim_param)

                for sym in parsed.free_symbols:
                    if clean_onnx_name(str(sym)) not in self.sdfg.symbols:
                        self.sdfg.add_symbol(clean_onnx_name(str(sym)),
                                             stype=int)
                    parsed = parsed.subs(
                        sym, dace.symbol(clean_onnx_name(str(sym))))

                shape.append(parsed)
            else:
                raise ValueError(
                    "Value '{}' does not have a shape in this graph."
                    " Please run shape inference before importing.".format(
                        name))
        transient = name not in self.inputs and name not in self.outputs
        if len(shape) == 0:
            self.sdfg.add_scalar(clean_onnx_name(name),
                                 dtype=onnx_tensor_type_to_typeclass(
                                     tensor_type.elem_type),
                                 transient=transient,
                                 storage=storage)
        else:
            self.sdfg.add_array(clean_onnx_name(name),
                                shape=shape,
                                dtype=onnx_tensor_type_to_typeclass(
                                    tensor_type.elem_type),
                                transient=transient,
                                storage=storage)

    @property
    def clean_weights(self):
        return {clean_onnx_name(k): v for k, v in self.weights.items()}

    def compile_and_init(self) -> compiled_sdfg.CompiledSDFG:
        """ Compile the SDFG and load parameters into GPU memory. """

        compiled_sdfg = self.sdfg.compile()

        # copy all parameters to the device
        self.initialized_parameters = {}
        for name, arr in self.weights.items():
            if clean_onnx_name(name) in compiled_sdfg.sdfg.arrays:
                desc = self.sdfg.arrays[clean_onnx_name(name)]
                cuda = is_cuda(desc.storage)
                if type(desc) is dt.Scalar:
                    self.initialized_parameters[clean_onnx_name(
                        name)] = arr.cuda() if cuda else arr.cpu().numpy()[()]
                else:
                    self.initialized_parameters[clean_onnx_name(
                        name)] = arr.cuda() if cuda else arr

        return compiled_sdfg

    def __call__(
        self, *args, **kwargs
    ) -> Union[Union[torch.Tensor, np.ndarray], Tuple[Union[torch.Tensor,
                                                            np.ndarray]]]:
        """ Execute the model.

            :param args: positional arguments to the model. The i-th argument will be passed as the i-th input of the
                         model.
            :param kwargs: named arguments to the model. The passed names should match the names in the ONNX model.
            :return: the output of the model (or a tuple of outputs if there are multiple).
        """

        transient_kwargs = {}
        if self.save_transients is not None:

            for node, parent in self.sdfg.all_nodes_recursive():
                if isinstance(node, nodes.AccessNode):
                    desc = self.sdfg.arrays[node.data]
                    if not isinstance(desc, dt.View) and desc.transient:
                        desc.transient = False
                        transient_kwargs[node.data] = desc

        if self.do_auto_optimize:
            self.auto_optimize()

        compiled = self.compile_and_init()

        inputs, symbols, outputs = self._call_args(args=args, kwargs=kwargs)

        for name, desc in transient_kwargs.items():
            transient_kwargs[name] = create_output_array(symbols,
                                                         desc,
                                                         use_torch=True,
                                                         zeros=False)
            self.save_transients[name] = transient_kwargs[name]

        compiled(**inputs, **outputs, **self.initialized_parameters, **symbols,
                 **transient_kwargs)

        if len(outputs) == 1:
            return next(iter(outputs.values()))

        return tuple(outputs.values())

    def _call_args(
        self,
        *,
        args,
        kwargs,
        torch_outputs: bool = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any], OrderedDict[str, Any]]:
        """ Prepare the arguments for a call.

            This returns 4 dicts; one for each of the following:
            1. the inputs
            3. inferred values for symbols for dynamic dimensions
            4. outputs

            These arguments can be passed to `self.sdfg`.

            :param args: model positional args
            :param kwargs: model kwargs
            :param torch_outputs: if not None, the outputs will be torch tensors depending on the boolean value.
                                  Otherwise the outputs will be torch tensors only if at least one of the inputs is a
                                  torch tensor.
            :return: the tuple of dicts
        """
        inputs = kwargs

        # convert the positional args to kwargs
        if len(args) > len(self.inputs):
            raise ValueError("Expected {} arguments, got {}".format(
                len(self.inputs), len(args)))

        inputs.update(dict(zip(self.inputs, args)))

        # check that there are no missing inputs
        if len(set(self.inputs).difference(inputs)) != 0:
            raise ValueError("Missing inputs {}".format(", ".join(
                set(self.inputs).difference(inputs))))

        # check that there are no unknown inputs
        # NOTE symbols can only be passed as kwargs
        if len(
                set(inputs).difference(self.inputs).difference(
                    self.sdfg.free_symbols)) != 0:
            raise ValueError("Unknown inputs {}".format(", ".join(
                set(inputs).difference(self.inputs))))

        clean_inputs = {}
        for input, arr in inputs.items():
            if input in self.sdfg.free_symbols:
                clean_inputs[input] = arr
            else:
                clean_inputs[clean_onnx_name(input)] = arr

        inferred_symbols = infer_symbols_from_shapes(self.sdfg, {
            **clean_inputs,
            **self.initialized_parameters
        })
        inferred_symbols = {k: int(v) for k, v in inferred_symbols.items()}

        if torch_outputs is None:
            torch_outputs = any(
                is_cuda(self.sdfg.arrays[clean_onnx_name(o)].storage)
                for o in self.outputs) or any(
                    isinstance(inp, torch.Tensor)
                    for _, inp in clean_inputs.items())

        outputs = collections.OrderedDict()
        # create numpy arrays for the outputs
        for output in self.outputs:
            clean_name = clean_onnx_name(output)
            outputs[clean_name] = create_output_array(
                inferred_symbols,
                self.sdfg.arrays[clean_name],
                use_torch=torch_outputs)

        # check that there's no overlap
        seen = set()
        for parameters in [
                clean_inputs, self.initialized_parameters, outputs,
                inferred_symbols
        ]:
            new_parameters = set(parameters)
            assert not seen.intersection(new_parameters)
            seen |= new_parameters

        return clean_inputs, inferred_symbols, outputs

    def expand_onnx_nodes(self):
        utils.expand_onnx_nodes(self.sdfg)

    def auto_optimize(self):
        utils.auto_optimize(
            self.sdfg,
            self.cuda,
            apply_strict=self.apply_strict,
            # constants have been folded before GPU transforms
            fold_constants=False)


def create_output_array(
        inferred_symbols: Dict[str, int],
        desc: dt.Data,
        use_torch=False,
        zeros: bool = False) -> Union[np.ndarray, torch.tensor]:
    """ Create the array for an output. This is either a numpy array or a torch tensor depending on `use_torch`

        When `self.force_torch_outputs` is True, the outputs will be tensors. Otherwise, the outputs will be tensors
        :param inferred_symbols: the symbols inferred from `infer_symbols_from_shapes`.
        :param desc: the data descriptor for the array
        :param use_torch: whether to return a numpy array or a torch tensor.
        :param zeros: if true init with zeros else empty.
    """
    def eval_dim(dim):
        for sym in dim.free_symbols:
            dim = dim.subs(sym, inferred_symbols[sym.name])
        return dim

    cuda = is_cuda(desc.storage)
    if cuda and not use_torch:
        raise ValueError("Got use_torch=False, but received a GPU descriptor")

    if isinstance(desc, dt.Scalar):
        shape = []
    else:
        shape = [
            eval_dim(d) if type(d) is dace.symbol else d for d in desc.shape
        ]

    if use_torch:
        # torch functions don't accept the empty shape, so create shape [1] then reshape to ()
        if len(shape) == 0:
            shape = [1]

        # as_numpy_dtype doesn't seem to work for indexing into the dict
        if desc.dtype == dace.pointer(dace.typeclass(None)):
            # assuming 64 bit ptrs
            dtype = torch.int64
        else:
            dtype = numpy_to_torch_dtype_dict[getattr(np,
                                                      desc.dtype.to_string())]
        tens = (torch.zeros if zeros else torch.empty)(*shape, dtype=dtype)
        if isinstance(desc, dt.Scalar):
            tens = tens.reshape(())

        return tens.cuda() if cuda else tens
    else:
        return (np.zeros if zeros else np.empty)(shape,
                                                 dtype=getattr(
                                                     np,
                                                     desc.dtype.to_string()))
