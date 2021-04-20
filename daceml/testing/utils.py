import numpy as np
import torch


def torch_tensors_close(name, torch_v, dace_v):
    """ Assert that the two torch tensors are close. Prints a nice error string if not.
    """
    rtol = 1e-6
    atol = 1e-4
    if not torch.allclose(
            torch_v, dace_v, rtol=rtol, atol=atol, equal_nan=True):
        print(f"torch value [{torch_v.shape}]: {torch_v}")
        print(f"dace value [{dace_v.shape}]: {dace_v}")
        print("diff: ", torch.abs(dace_v - torch_v))
        torch_v = torch_v.detach().cpu().numpy()
        dace_v = dace_v.detach().cpu().numpy()
        failed_mask = np.abs(torch_v - dace_v) > atol + rtol * np.abs(dace_v)

        if failed_mask.sum() < 100:
            print(f"wrong elements torch: {torch_v[failed_mask]}")
            print(f"wrong elements dace: {dace_v[failed_mask]}")
            for x, y in zip(torch_v[failed_mask], dace_v[failed_mask]):
                print(f"lhs_failed: {abs(x - y)}")
                print(f"rhs_failed: {atol} + {rtol * abs(y)}")

        assert False, f"{name} was not close)"
