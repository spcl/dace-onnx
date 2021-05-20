import numpy as np
import torch
import typing


def torch_tensors_close(name, torch_v, dace_v):
    """ Assert that the two torch tensors are close. Prints a nice error string if not.
    """
    rtol = 1e-5
    atol = 1e-4
    if not torch.allclose(
            torch_v, dace_v, rtol=rtol, atol=atol, equal_nan=True):
        print(name + " was not close")
        print("torch value: ", torch_v)
        print("dace value: ", dace_v)
        print("diff: ", torch.abs(dace_v - torch_v))

        failed_mask = np.abs(torch_v.detach().cpu().numpy() - dace_v.detach(
        ).cpu().numpy()) > atol + rtol * np.abs(dace_v.detach().cpu().numpy())
        print(f"wrong elements torch: {torch_v[failed_mask]}")
        print(f"wrong elements dace: {dace_v[failed_mask]}")

        for x, y, _ in zip(torch_v[failed_mask], dace_v[failed_mask],
                           range(100)):
            print(f"lhs_failed: {abs(x - y)}")
            print(
                f"rhs_failed: {atol} + {rtol * abs(y)} = {atol + rtol * abs(y)}"
            )

        assert False, f"{name} was not close)"


T = typing.TypeVar("T")


def copy_to_gpu(gpu: bool, tensor: T) -> T:
    if gpu:
        return tensor.cuda()
    else:
        return tensor
