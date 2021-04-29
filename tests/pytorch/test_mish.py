import dace
import pytest
import torch

from daceml.pytorch import dace_module, DaceModule
from daceml.testing import torch_tensors_close


class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


@pytest.mark.pure
def test_mish():
    pt_model = Mish()
    dace_model = DaceModule(Mish(), cuda=True, backward=True)

    pt_inputs = torch.rand(8, 32, 608, 608).cuda()
    dace_inputs = torch.clone(pt_inputs)
    pt_inputs.requires_grad=True
    dace_inputs.requires_grad=True

    dy = torch.rand(8, 32, 608, 608).cuda()

    pt_out = pt_model(pt_inputs)
    dace_out = dace_model(dace_inputs)

    torch_tensors_close("output", pt_out, dace_out)
    pt_out.backward(dy)
    dace_out.backward(dy)
    torch_tensors_close("grad", pt_inputs.grad, dace_inputs.grad)


