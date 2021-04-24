import numpy as np
import torch


def torch_tensors_close(name, torch_v, dace_v):
    """ Assert that the two torch tensors are close. Prints a nice error string if not.
    """
    rtol = 1e-6
    atol = 1e-4
    if not torch.allclose(
            torch_v, dace_v, rtol=rtol, atol=atol, equal_nan=True):
        torch_v = torch_v.detach().cpu().numpy()
        dace_v = dace_v.detach().cpu().numpy()
        failed_mask = np.abs(torch_v - dace_v) > atol + rtol * np.abs(dace_v)

        for x, y, _ in zip(torch_v[failed_mask], dace_v[failed_mask],
                           range(100)):
            print(f"{name} was not close:")
            print(
                f"{x:.4f}\t\t!= {y:.4f}\t\t {abs(x - y)} <!= {atol} + {rtol * abs(y)}"
            )

        assert False, f"{name} was not close)"
