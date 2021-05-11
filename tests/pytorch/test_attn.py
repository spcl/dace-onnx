import torch
import numpy as np
import pytest

from daceml.pytorch import DaceModule

from dace.transformation.dataflow import RedundantSecondArray

from daceml.testing import copy_to_gpu, torch_tensors_close
from daceml.transformation import ConstantFolding


@pytest.mark.ort
def test_attn(gpu, sdfg_name):
    B = 2
    H = 16
    P = 64
    N = P * H
    SM, SN = 512, 512
    K, Q, V = [
        copy_to_gpu(gpu, torch.randn([SM, B, N])),
        copy_to_gpu(gpu, torch.randn([SN, B, N])),
        copy_to_gpu(gpu, torch.randn([SM, B, N]))
    ]
    ptmodel = copy_to_gpu(gpu, torch.nn.MultiheadAttention(N, H, bias=False))

    pt_outputs = ptmodel(Q, K, V)

    dace_model = DaceModule(ptmodel, sdfg_name=sdfg_name)
    dace_outputs_0 = dace_model(Q, K, V)

    dace_model.dace_model.sdfg.apply_transformations_repeated(
        [ConstantFolding, RedundantSecondArray],
        validate_all=True,
        strict=True)
    dace_outputs_1 = dace_model(Q, K, V)

    torch_tensors_close("outputs_0", pt_outputs[0], dace_outputs_1[0])
    torch_tensors_close("outputs_1", pt_outputs[1], dace_outputs_1[1])
