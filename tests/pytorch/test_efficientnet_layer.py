import copy
import itertools
import math
from contextlib import suppress

import pytest
import numpy as np

import torch
from dace.library import change_default
from efficientnet_pytorch.model import MBConvBlock, get_model_params

from dace.transformation.dataflow import RedundantSecondArray
from torch import nn
from torch.nn import functional as F

import daceml.onnx as donnx
from daceml.pytorch import DaceModule
from daceml.testing.utils import torch_tensors_close
from daceml.transformation import ConstantFolding, ConstantDeviceCopyElimination, parameter_to_transient


def test_cudnn_conv():
    nn.Parameter
    donnx.ONNXConv.default_implementation = "cuDNN"
    inputs = torch.rand(1, 32, 224, 224)

    pt_model = nn.Conv2d(32, 8, kernel_size=(3, 3))

    dace_model = DaceModule(pt_model, cuda=True)
    dace_model.append_post_onnx_hook("view", lambda m: m.sdfg.view())
    out = dace_model(inputs)
    pt_out = pt_model(inputs)
    torch_tensors_close("output", pt_out, out)


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


# @pytest.mark.pure
# def test_bn(gpu, sdfg_name):
#     with change_default(donnx.ONNXBatchNormalization,
#                         "cuDNN") if gpu else suppress():
#         torch.random.manual_seed(42)
#         inputs = torch.rand(1, 64, 60, 60)
#
#         pt_model = nn.BatchNorm2d(64)
#         dace_model = nn.BatchNorm2d(64)
#
#         dace_model.load_state_dict(pt_model.state_dict())
#
#         dace_model = DaceModule(dace_model, cuda=gpu, sdfg_name=sdfg_name)
#         if gpu:
#
#             def param_to_trans(model):
#                 for name, _ in itertools.chain(
#                         model.pytorch_model.named_buffers(),
#                         model.pytorch_model.named_parameters()):
#                     if name != "num_batches_tracked":
#                         parameter_to_transient(model, name)
#
#             dace_model.append_post_onnx_hook("param_to_transient",
#                                              param_to_trans)
#         dace_model.append_post_onnx_hook("view", lambda m: m.sdfg.view())
#         dace_output = dace_model(inputs.cuda())
#         pt_output = pt_model(inputs)
#
#         torch_tensors_close("output", pt_output, dace_output.cpu())
#         torch_tensors_close("mean", pt_model.running_mean,
#                             dace_model.pytorch_model.running_mean.cpu())
#         torch_tensors_close("var", pt_model.running_var,
#                             dace_model.pytorch_model.running_var.cpu())


@pytest.mark.pure
@pytest.mark.gpu
def test_bn_training(sdfg_name):
    with change_default(donnx.ONNXBatchNormalization, "cuDNN"):
        torch.random.manual_seed(42)
        inputs = torch.rand(1, 64, 60, 60, requires_grad=True).cuda()
        dy = torch.rand(1, 64, 60, 60).cuda()

        pt_model = nn.BatchNorm2d(64).cuda()
        dace_model = nn.BatchNorm2d(64).cuda()

        dace_model.load_state_dict(pt_model.state_dict())

        dace_model = DaceModule(dace_model,
                                cuda=True,
                                sdfg_name=sdfg_name,
                                backward=True)

        def param_to_trans(model):
            for name, _ in itertools.chain(
                    model.pytorch_model.named_buffers(),
                    model.pytorch_model.named_parameters()):
                if name != "num_batches_tracked":
                    parameter_to_transient(model, name)

        dace_model.append_post_onnx_hook("param_to_transient", param_to_trans)
        dace_model.append_post_onnx_hook("view", lambda m: m.sdfg.view())
        pt_output = pt_model(inputs)
        dace_output = dace_model(inputs)

        torch_tensors_close("output", pt_output, dace_output)
        torch_tensors_close("mean", pt_model.running_mean,
                            dace_model.pytorch_model.running_mean)
        torch_tensors_close("var", pt_model.running_var,
                            dace_model.pytorch_model.running_var)
        pt_output.backward(dy)
        dace_output.backward(dy)

        for (pt_name, pt_buf), (dace_name,
                                dace_buf) in zip(pt_model.named_buffers(),
                                                 dace_model.named_buffers()):
            assert pt_name in dace_name
            if "num_batches_tracked" not in pt_name:
                torch_tensors_close(pt_name, pt_buf, dace_buf)

        for (name,
             dace_param), (pt_name,
                           pt_param) in zip(pt_model.named_parameters(),
                                            dace_model.named_parameters()):
            assert 'pytorch_model.' + name == pt_name
            torch_tensors_close(name, pt_param.detach(), dace_param.detach())


@pytest.mark.pure
@pytest.mark.gpu
def test_mbconv_training(sdfg_name):
    with change_default(donnx.ONNXBatchNormalization, "cuDNN"), \
         change_default(donnx.ONNXConv, "cuDNN"):
        torch.random.manual_seed(42)
        inputs = torch.rand(1, 64, 60, 60, requires_grad=True).cuda()
        dy = torch.rand(1, 64, 60, 60).cuda()

        block_params, global_params = get_model_params("efficientnet-b0", {})

        pt_model = MBConvBlock(block_params[0], global_params).cuda()
        pt_model.set_swish(memory_efficient=False)
        dace_model = MBConvBlock(block_params[0], global_params).cuda()
        dace_model.set_swish(memory_efficient=False)

        dace_model.load_state_dict(pt_model.state_dict())

        dace_model = DaceModule(dace_model, cuda=True, backward=True)
        dace_model.prepend_post_onnx_hook(
            "cf",
            lambda onnx_model: onnx_model.sdfg.apply_transformations_repeated(
                {
                    ConstantFolding, RedundantSecondArray,
                    ConstantDeviceCopyElimination
                },
                validate_all=True,
                strict=True))

        pt_output = pt_model(inputs)
        dace_output = dace_model(inputs)

        torch_tensors_close("output", pt_output, dace_output)
        pt_output.backward(dy)
        dace_output.backward(dy)

        for (name,
             dace_param), (pt_name,
                           pt_param) in zip(pt_model.named_parameters(),
                                            dace_model.named_parameters()):
            assert 'pytorch_model.' + name == pt_name
            torch_tensors_close(name, pt_param.detach(), dace_param.detach())


@pytest.mark.pure
def test_mbconv():
    donnx.ONNXConv.default_implementation = "cuDNN"
    donnx.ONNXBatchNormalization.default_implementation = "cuDNN"
    inputs = torch.rand(8, 32, 224, 224).cuda()
    dy = torch.rand(8, 16, 224, 224).cuda()

    block_params, global_params = get_model_params("efficientnet-b0", {})

    pt_model = MBConvBlock(block_params[0], global_params).cuda()
    pt_model.set_swish(memory_efficient=False)
    dace_model = MBConvBlock(block_params[0], global_params).cuda()
    dace_model.set_swish(memory_efficient=False)

    dace_model.load_state_dict(pt_model.state_dict())

    dace_model = DaceModule(dace_model, cuda=True, backward=True)
    dace_model.prepend_post_onnx_hook(
        "cf",
        lambda onnx_model: onnx_model.sdfg.apply_transformations_repeated(
            {
                ConstantFolding, RedundantSecondArray,
                ConstantDeviceCopyElimination
            },
            validate_all=True,
            strict=True))

    def param_to_trans(model):
        for name, _ in itertools.chain(model.pytorch_model.named_buffers(),
                                       model.pytorch_model.named_parameters()):
            if "num_batches_tracked" not in name:
                parameter_to_transient(model, name)

    dace_model.append_post_onnx_hook("param_to_transient", param_to_trans)

    dace_output = dace_model(inputs)
    pt_output = pt_model(inputs)

    torch_tensors_close("output", pt_output, dace_output)

    for (pt_name, pt_buf), (dace_name,
                            dace_buf) in zip(pt_model.named_buffers(),
                                             dace_model.named_buffers()):
        assert pt_name in dace_name
        if "num_batches_tracked" not in pt_name:
            torch_tensors_close(pt_name, pt_buf, dace_buf)

    pt_model.backward(dy)
    dace_model.backward(dy)
