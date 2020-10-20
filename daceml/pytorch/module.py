import logging
import os
import tempfile
from functools import wraps
import typing

import torch
import torch.nn as nn
import onnx
from torch.onnx import TrainingMode

from daceml.onnx import ONNXModel
from daceml.onnx.shape_inference import infer_shapes


class DaceModule(nn.Module):
    """ A wrapper that converts a PyTorch ``nn.Module`` to a PyTorch compatible data-centric ``nn.Module``.

        :param module: the model to wrap.
        :param dummy_inputs: a tuple of tensors to use as input when tracing ``model``.
        :param cuda: if ``True``, the module will execute using CUDA.
        :param train: whether to use train mode when tracing ``model``.
        :param apply_strict: whether to apply strict transforms after conversion (this generally improves performance,
                             but can be slow).

        :Example:

            >>> from daceml.pytorch import DaceModule
            >>> class MyModule(nn.Module):
            ...     def forward(self, x):
            ...        x = torch.log(x)
            ...        x = torch.sqrt(x)
            ...        return x
            >>> module = MyModule()
            >>> module(torch.ones(2))
            tensor([0., 0.])
            >>> dace_module = DaceModule(module)
            >>> dace_module(torch.ones(2))
            Automatically expanded library node "ONNX_Log_0".
            Automatically expanded library node "ONNX_Sqrt_1".
            array([0., 0.], dtype=float32)
    """
    def __init__(
        self,
        module: nn.Module,
        dummy_inputs: typing.Optional[typing.Tuple[torch.Tensor]] = None,
        cuda: bool = False,
        train: bool = False,
        apply_strict: bool=False):
        super(DaceModule, self).__init__()

        self.model = module
        self.train = train
        self.sdfg = None
        self.cuda = cuda
        self.apply_strict = apply_strict
        if dummy_inputs is not None:
            self.dace_model = self._initialize_sdfg(dummy_inputs)

    def _initialize_sdfg(self, dummy_inputs) -> ONNXModel:
        with tempfile.TemporaryDirectory() as dir_name:
            export_name = os.path.join(dir_name, "export.onnx")

            torch.onnx.export(self.model,
                              dummy_inputs,
                              export_name,
                              verbose=logging.root.level <= logging.DEBUG,
                              training=(TrainingMode.TRAINING
                                        if self.train else TrainingMode.EVAL),
                              opset_version=12)

            onnx_model = infer_shapes(onnx.load(export_name))
            self.onnx_model = onnx_model

            dace_model = ONNXModel("dace_model",
                                   onnx_model,
                                   infer_shapes=False,
                                   cuda=self.cuda,
                                   apply_strict=self.apply_strict)
            self.sdfg = dace_model.sdfg
            self.sdfg.validate()

            return dace_model

    def forward(self, *actual_inputs):
        """ Execute the forward pass using the traced ``module``."""
        if self.sdfg is None:
            self.dace_model = self._initialize_sdfg(actual_inputs)

        return self.dace_model(*actual_inputs)


def dace_module(moduleclass):
    """ Decorator to apply on a definition of a ``torch.nn.Module`` to
        convert it to a data-centric module upon construction.

        :Example:

            >>> from daceml.pytorch import dace_module
            >>> @dace_module
            ... class MyModule(nn.Module):
            ...     def forward(self, x):
            ...        x = torch.log(x)
            ...        x = torch.sqrt(x)
            ...        return x
            >>> module = MyModule()
            >>> module(torch.ones(2))
            Automatically expanded library node "ONNX_Log_0".
            Automatically expanded library node "ONNX_Sqrt_1".
            array([0., 0.], dtype=float32)
    """
    @wraps(moduleclass)
    def _create(*args, **kwargs):
        return DaceModule(moduleclass(*args, **kwargs))

    return _create
