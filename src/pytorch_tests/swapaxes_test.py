# Owner(s): ["module: tests"]
import torch
import numpy as np

import unittest
from itertools import product, permutations, combinations
from functools import partial
import random

from torch.testing import make_tensor
from torch.testing._internal.common_utils import (
    IS_FBCODE, TestCase, run_tests, suppress_warnings, gradcheck, gradgradcheck,
    numpy_to_torch_dtype_dict
)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, onlyCPU, dtypes, onlyNativeDeviceTypes, skipMeta)
from torch.testing._internal.common_dtype import (
    all_types_and_complex_and, complex_types, all_types_and, floating_and_complex_types_and,
)

# TODO: replace this with make_tensor() in common_utils.py
def _generate_input(shape, dtype, device, with_extremal):
    if shape == ():
        x = torch.tensor((), dtype=dtype, device=device)
    else:
        if dtype.is_floating_point or dtype.is_complex:
            # work around torch.randn not being implemented for bfloat16
            if dtype == torch.bfloat16:
                x = torch.randn(*shape, device=device) * random.randint(30, 100)
                x = x.to(torch.bfloat16)
            else:
                x = torch.randn(*shape, dtype=dtype, device=device) * random.randint(30, 100)
            x[torch.randn(*shape) > 0.5] = 0
            if with_extremal and dtype.is_floating_point:
                # Use extremal values
                x[torch.randn(*shape) > 0.5] = float('nan')
                x[torch.randn(*shape) > 0.5] = float('inf')
                x[torch.randn(*shape) > 0.5] = float('-inf')
            elif with_extremal and dtype.is_complex:
                x[torch.randn(*shape) > 0.5] = complex('nan')
                x[torch.randn(*shape) > 0.5] = complex('inf')
                x[torch.randn(*shape) > 0.5] = complex('-inf')
        elif dtype == torch.bool:
            x = torch.zeros(shape, dtype=dtype, device=device)
            x[torch.randn(*shape) > 0.5] = True
        else:
            x = torch.randint(15, 100, shape, dtype=dtype, device=device)

    return x

# TODO: replace this with make_tensor() in common_utils.py
def _rand_shape(dim, min_size, max_size):
    shape = []
    for i in range(dim):
        shape.append(random.randint(min_size, max_size))
    return tuple(shape)

# TODO: refactor tests to avoid this function
# Converts half/bfloat16 dtype to float when device is cpu
def _convert_t(dtype, device):
    if device == 'cpu' and dtype in {torch.half, torch.bfloat16}:
        return torch.float
    return dtype

# TODO: replace this with make_tensor() in common_utils.py
# Returns a tensor of the requested shape, dtype, and device
# Requesting a half CPU tensor returns a float CPU tensor with
# values representable by a half.
# Initialization uses randint for non-float types and randn for float types.
def _make_tensor(shape, dtype, device, fill_ones=False) -> torch.Tensor:
    # Returns a tensor filled with ones
    if fill_ones:
        return torch.ones(*shape, dtype=_convert_t(dtype, device), device=device)

    # Returns a tensor with random integer values
    if not (dtype.is_floating_point or dtype.is_complex):
        t = torch.randint(0, 10, shape, device=device)
        if dtype != torch.uint8:
            t = t - 5  # generate negative values also
        return t.to(_convert_t(dtype, device))

    # Populates the CPU tensor with floats representable as half/bfloat16
    if dtype == torch.half and device == 'cpu':
        return torch.randn(*shape, dtype=torch.float, device=device).half().float()
    if dtype == torch.bfloat16 and device == 'cpu':
        return torch.randn(*shape, dtype=torch.float, device=device).bfloat16().float()

    # Default: returns a tensor with random float values
    return torch.randn(shape, dtype=dtype, device=device).to(dtype=dtype)



class TestOldViewOps(TestCase):

    @onlyNativeDeviceTypes
    @dtypes(torch.int64, torch.float, torch.complex128)
    def test_transpose_invalid(self, device, dtype):
        for fn in (torch.swapdims, torch.swapaxes, torch.transpose):
            shape = _rand_shape(4, min_size=5, max_size=10)
            x = _generate_input(shape, dtype, device, False)

            # Invalid `source` and `destination` dimension
            with self.assertRaisesRegex(IndexError, "Dimension out of range"):
                fn(x, 5, 0)

            with self.assertRaisesRegex(IndexError, "Dimension out of range"):
                fn(x, 0, 5)

    @dtypes(torch.int64, torch.float, torch.complex128)
    def test_transpose_vs_numpy(self, device, dtype):
        for fn in (torch.swapdims, torch.swapaxes, torch.transpose):
            for nd in range(5):
                shape = _rand_shape(nd, min_size=5, max_size=10)
                x = _generate_input(shape, dtype, device, with_extremal=False)
                for random_negative in [True, False]:
                    for src_dim, dst_dim in permutations(range(nd), r=2):
                        random_prob = random.random()

                        if random_negative and random_prob > 0.66:
                            src_dim = src_dim - nd
                        elif random_negative and random_prob > 0.33:
                            dst_dim = dst_dim - nd
                        elif random_negative:
                            src_dim = src_dim - nd
                            dst_dim = dst_dim - nd

                        partial_map = {
                            torch.swapdims: partial(torch.swapdims, dim0=src_dim, dim1=dst_dim),
                            torch.swapaxes: partial(torch.swapaxes, axis0=src_dim, axis1=dst_dim),
                            torch.transpose: partial(torch.transpose, dim0=src_dim, dim1=dst_dim),
                        }

                        torch_fn = partial_map[fn]
                        np_fn = partial(np.swapaxes, axis1=src_dim, axis2=dst_dim)
                        self.compare_with_numpy(torch_fn, np_fn, x, device=None, dtype=None)

            # Move dim to same position
            x = torch.randn(2, 3, 5, 7, 11)
            partial_map = {
                torch.swapdims: partial(torch.swapdims, dim0=0, dim1=0),
                torch.swapaxes: partial(torch.swapaxes, axis0=0, axis1=0),
                torch.transpose: partial(torch.transpose, dim0=0, dim1=0),
            }
            torch_fn = partial_map[fn]
            np_fn = partial(np.swapaxes, axis1=0, axis2=0)
            self.compare_with_numpy(torch_fn, np_fn, x, device=None, dtype=None)

instantiate_device_type_tests(TestOldViewOps, globals())

if __name__ == '__main__':
    run_tests()
