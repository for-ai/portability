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
    numpy_to_torch_dtype_dict, 
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

    @onlyCPU
    @dtypes(torch.float)
    def test_broadcast_tensors(self, device, dtype):
        x0 = torch.randn(2, 1, 3, dtype=dtype, device=device)
        x1 = torch.randn(3, dtype=dtype, device=device)
        x2 = torch.randn(3, 1, dtype=dtype, device=device)
        expected_size = (2, 3, 3)

        y0, y1, y2 = torch.broadcast_tensors(x0, x1, x2)
        self.assertTrue(y0.size() == expected_size)
        self.assertTrue(y1.size() == expected_size)
        self.assertTrue(y2.size() == expected_size)


    @onlyCPU
    def test_broadcast_shapes(self, device):
        examples = [(), (1,), (2,), (1, 1), (3, 1), (3, 2), (4, 1, 1), (4, 3, 2)]
        for s0 in examples:
            x0 = torch.randn(s0)
            expected = torch.broadcast_tensors(x0)[0].shape
            actual = torch.broadcast_shapes(s0)
            self.assertEqual(expected, actual)

            for s1 in examples:
                x1 = torch.randn(s1)
                expected = torch.broadcast_tensors(x0, x1)[0].shape
                actual = torch.broadcast_shapes(s0, s1)
                self.assertEqual(expected, actual)

        inputs_list = [[1, 4], [4, 1], [1, 1, 3]]
        for integral_inputs in inputs_list:
            res1 = torch.broadcast_shapes(*integral_inputs)
            res2 = torch.broadcast_tensors(*map(torch.empty, integral_inputs))[0].shape
            self.assertEqual(res1, res2)

        inputs_with_neg_vals = [[1, 1, -12], [-1, 1], [-11, ]]
        for integral_inputs_with_neg_vals in inputs_with_neg_vals:
            with self.assertRaisesRegex(RuntimeError, "Trying to create tensor with negative dimension"):
                torch.broadcast_shapes(*integral_inputs_with_neg_vals)

        integral_inputs_error_case = [(3, 5), (2, 4, 1)]
        for error_input in integral_inputs_error_case:
            with self.assertRaisesRegex(RuntimeError, "Shape mismatch: objects cannot be broadcast to a single shape"):
                torch.broadcast_shapes(*error_input)

        negative_inputs = [(-1,), (1, -12), (4, -11), (-4, 1), (1, 1, -2)]
        for s0 in negative_inputs:
            with self.assertRaisesRegex(RuntimeError, "Trying to create tensor with negative dimension"):
                torch.broadcast_shapes(s0)

            for s1 in negative_inputs:
                with self.assertRaisesRegex(RuntimeError, "Trying to create tensor with negative dimension"):
                    torch.broadcast_shapes(s0, s1)

        float_inputs_error_case = [(1.1, 2.0), (1.1, 1.0)]
        for error_case in float_inputs_error_case:
            for float_input in error_case:
                with self.assertRaisesRegex(RuntimeError, "Input shapes "
                                            "should be of type ints, a tuple of ints, or a list of ints"):
                    torch.broadcast_shapes(float_input)

        diff_input_types = [(1, (5,)), (3, (1,)), (1, (3, 4))]
        for s0 in diff_input_types:
            res1 = torch.broadcast_shapes(*s0)
            res2 = torch.broadcast_tensors(*map(torch.empty, s0))[0].shape
            self.assertEqual(res1, res2)

instantiate_device_type_tests(TestOldViewOps, globals())

if __name__ == '__main__':
    run_tests()
