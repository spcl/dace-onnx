import copy
import math

import pytest
import numpy as np

import torch
from efficientnet_pytorch.model import MBConvBlock, get_model_params

from dace.transformation.dataflow import RedundantSecondArray
from torch import nn
from torch.nn import functional as F

from daceml.pytorch import DaceModule
from daceml.testing.utils import torch_tensors_close
from daceml.transformation import ConstantFolding


class Conv2dDynamicSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow, for a dynamic image size.
       The padding is operated in forward function by calculating dynamically.
    """

    # Tips for 'SAME' mode padding.
    #     Given the following:
    #         i: width or height
    #         s: stride
    #         k: kernel size
    #         d: dilation
    #         p: padding
    #     Output after Conv2d:
    #         o = floor((i+p-((k-1)*d+1))/s+1)
    # If o equals i, i = floor((i+p-((k-1)*d+1))/s+1),
    # => p = (i-1)*s+((k-1)*d+1)-i

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0,
                         dilation, groups, bias)
        self.stride = self.stride if len(
            self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(
            iw / sw)  # change the output size according to stride ! ! !
        pad_h = max(
            (oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih,
            0)
        pad_w = max(
            (ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw,
            0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [
                pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2
            ])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def testconv2dynpad():
    inputs = torch.rand(1, 32, 224, 224)

    pt_model = nn.BatchNorm2d(num_features=32)

    dace_model = DaceModule(pt_model)
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

    # torch_tensors_close("X_minus_mean", torch.transients["ONNX_BatchNormalization_0_expansion_X_minus_mean"], torch.from_numpy(X_minus_mean))

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

    block_params, global_params = get_model_params("efficientnet-b0", {})

    pt_model = nn.BatchNorm2d(64)
    dace_model = nn.BatchNorm2d(64)

    dace_model.load_state_dict(pt_model.state_dict())

    dace_model = DaceModule(dace_model, cuda=gpu, sdfg_name=sdfg_name)
    dace_output = dace_model(inputs)
    pt_output = pt_model(inputs)

    #output = np.empty_like(inputs.numpy())
    #svm = np.empty((64,), dtype=np.float32)
    #svv = np.empty((64,), dtype=np.float32)

    #bn_numpy(inputs.detach().numpy(), pt_model.weight.detach().numpy(), pt_model.bias.detach().numpy(), None, None, output, None, None, svm , svv, pt_model.eps)

    # torch_tensors_close("np_output", torch.from_numpy(output), pt_output)
    torch_tensors_close("output", pt_output, dace_output)
    torch_tensors_close("mean", pt_model.running_mean,
                        dace_model.pytorch_model.running_mean)
    torch_tensors_close("var", pt_model.running_var,
                        dace_model.pytorch_model.running_var)


@pytest.mark.ort
def test_mbconv():
    inputs = torch.rand(1, 32, 224, 224)

    block_params, global_params = get_model_params("efficientnet-b0", {})

    pt_model = MBConvBlock(block_params[0], global_params)
    pt_model.set_swish(memory_efficient=False)
    dace_model = MBConvBlock(block_params[0], global_params)
    dace_model.set_swish(memory_efficient=False)

    dace_model.load_state_dict(pt_model.state_dict())

    dace_model = DaceModule(dace_model, cuda=False)
    dace_model.prepend_post_onnx_hook(
        "cf",
        lambda onnx_model: onnx_model.sdfg.apply_transformations_repeated(
            {ConstantFolding, RedundantSecondArray},
            validate_all=True,
            strict=True))
    dace_model.prepend_post_onnx_hook("view", lambda m: m.sdfg.view())
    dace_model.append_post_onnx_hook("view", lambda m: m.sdfg.view())
    dace_model.append_post_onnx_hook(
        "save_transients", lambda dace_module: exec(
            "dace_module.dace_onnx_model.save_transients = {}"))

    dace_output = dace_model(inputs)
    torch.__dict__["transients"] = dace_model.dace_onnx_model.save_transients
    torch.__dict__["dace_model"] = dace_model
    pt_output = pt_model(inputs)

    torch_tensors_close("output", pt_output, dace_output)
