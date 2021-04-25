import copy
import math

import pytest
import numpy as np

import torch
from efficientnet_pytorch.model import MBConvBlock, get_model_params

from dace.transformation.dataflow import RedundantSecondArray
from torch import nn
from torch.nn import functional as F

import daceml.onnx as donnx
from daceml.pytorch import DaceModule
from daceml.testing.utils import torch_tensors_close
from daceml.transformation import ConstantFolding, ConstantDeviceCopyElimination


def test_cudnn_conv():
    donnx.ONNXConv.default_implementation = "cuDNN"
    inputs = torch.rand(1, 32, 224, 224)

    pt_model = nn.Conv2d(32, 8, kernel_size=(3, 3), bias=False)

    dace_model = DaceModule(pt_model, cuda=True)
    dace_model.append_post_onnx_hook("view", lambda m: m.sdfg.view())
    dace_model(inputs)


def bn_numpy(X, scale, B, in_mean, in_var, Y, out_mean, out_var, saved_mean,
             saved_var, eps):
    reduce_axes = list(copy.deepcopy(X.shape))
    num_channels = reduce_axes.pop(1)
    broadcast_shape = [num_channels, 1, 1]
    N = reduce_axes[0] * reduce_axes[1] * reduce_axes[2]
    saved_mean[:] = np.add.reduce(X, axis=(0, 2, 3)) / N

    saved_mean_broadcastable = saved_mean.reshape((-1, 1, 1))

    X_minus_mean = (X - saved_mean_broadcastable)

    saved_var[:] = np.add.reduce(X_minus_mean * X_minus_mean,
                                 axis=(0, 2, 3)) / N
    saved_var_eps = np.reshape(saved_var + eps, broadcast_shape)

    normalized = X_minus_mean / (np.sqrt(saved_var_eps))

    scale_reshaped = np.reshape(scale, broadcast_shape)
    bias_reshaped = np.reshape(B, broadcast_shape)
    Y[:] = normalized * scale_reshaped + bias_reshaped


@pytest.mark.pure
def test_bn(gpu, sdfg_name):
    torch.random.manual_seed(42)
    inputs = torch.rand(1, 64, 60, 60)

    pt_model = nn.BatchNorm2d(64)
    dace_model = nn.BatchNorm2d(64)

    dace_model.load_state_dict(pt_model.state_dict())

    dace_model = DaceModule(dace_model, cuda=False, sdfg_name=sdfg_name)
    dace_output = dace_model(inputs)
    pt_output = pt_model(inputs)

    torch_tensors_close("output", pt_output, dace_output)
    torch_tensors_close("mean", pt_model.running_mean,
                        dace_model.pytorch_model.running_mean)
    torch_tensors_close("var", pt_model.running_var,
                        dace_model.pytorch_model.running_var)


@pytest.mark.pure
def test_mbconv(gpu):
    inputs = torch.rand(1, 32, 224, 224)

    block_params, global_params = get_model_params("efficientnet-b0", {})

    pt_model = MBConvBlock(block_params[0], global_params)
    pt_model.set_swish(memory_efficient=False)
    dace_model = MBConvBlock(block_params[0], global_params)
    dace_model.set_swish(memory_efficient=False)

    dace_model.load_state_dict(pt_model.state_dict())

    dace_model = DaceModule(dace_model, cuda=gpu)
    dace_model.prepend_post_onnx_hook(
        "cf",
        lambda onnx_model: onnx_model.sdfg.apply_transformations_repeated(
            {
                ConstantFolding, RedundantSecondArray,
                ConstantDeviceCopyElimination
            },
            validate_all=True,
            strict=True))

    dace_output = dace_model(inputs)
    pt_output = pt_model(inputs)

    torch_tensors_close("output", pt_output, dace_output)
