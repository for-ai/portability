# Owner(s): ["module: tensor creation"]

import torch
import numpy as np

import sys
import math
import warnings
import unittest
from itertools import product, combinations, combinations_with_replacement, permutations
import random

from torch.testing import make_tensor
from torch.testing._internal.common_utils import (
    TestCase, run_tests, do_test_empty_full, TEST_WITH_ROCM, suppress_warnings,
    torch_to_numpy_dtype_dict, numpy_to_torch_dtype_dict, slowTest,
    TEST_SCIPY, IS_MACOS, IS_PPC, IS_WINDOWS, parametrize)
from torch.testing._internal.common_device_type import (
    expectedFailureMeta, instantiate_device_type_tests, deviceCountAtLeast, onlyNativeDeviceTypes,
    onlyCPU, largeTensorTest, precisionOverride, dtypes,
    onlyCUDA, skipCPUIf, dtypesIfCUDA, skipMeta)
from torch.testing._internal.common_dtype import (
    all_types_and_complex_and, all_types_and, floating_and_complex_types,
    floating_types, floating_and_complex_types_and, integral_types_and, get_all_dtypes
)
from torch.testing._creation import float_to_corresponding_complex_type_map
from ..utils.pytorch_device_decorators import onlyNativeDeviceTypes, onlyAcceleratedDeviceTypes, instantiate_device_type_tests
from ..utils.timer_wrapper import pytorch_op_timer

from torch.utils.dlpack import to_dlpack

# TODO: replace with make_tensor


def _generate_input(shape, dtype, device, with_extremal):
    if shape == ():
        x = torch.tensor((), dtype=dtype, device=device)
    else:
        if dtype.is_floating_point or dtype.is_complex:
            # work around torch.randn not being implemented for bfloat16
            if dtype == torch.bfloat16:
                x = torch.randn(*shape, device=device) * \
                    random.randint(30, 100)
                x = x.to(torch.bfloat16)
            else:
                x = torch.randn(*shape, dtype=dtype,
                                device=device) * random.randint(30, 100)
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


# TODO: replace with make_tensor
def _rand_shape(dim, min_size, max_size):
    shape = []
    for i in range(dim):
        shape.append(random.randint(min_size, max_size))
    return tuple(shape)

# Test suite for tensor creation ops
#
# Includes creation functions like torch.eye, random creation functions like
#   torch.rand, and *like functions like torch.ones_like.
# DOES NOT INCLUDE view ops, which are tested in TestViewOps (currently in
#   test_torch.py) OR numpy interop (which is also still tested in test_torch.py)
#
# See https://pytorch.org/docs/master/torch.html#creation-ops


class TestRandomTensorCreation(TestCase):
    exact_dtype = True

    def test_randperm(self, device):
        if device == 'cpu' or device == 'meta' or device == 'xla:1':
            rng_device = None
        else:
            # TODO: This won't actually work for non-CUDA device
            # see https://github.com/pytorch/pytorch/issues/54282
            rng_device = [device]

        # Test core functionality. On CUDA, different value of n has different
        # code path
        for n in (5, 100, 50000, 100000):
            # Ensure both integer and floating-point numbers are tested. Half follows an execution path that is
            # different from others on CUDA.
            for dtype in (torch.long, torch.half, torch.float, torch.bfloat16):
                if dtype != torch.half or device != 'xla:1':
                    # Large n for torch.half will raise an exception, do not test here.
                    if n > 2049 and dtype == torch.half:
                        continue
                    if dtype == torch.bfloat16 and device != 'cpu':
                        continue
                    if n > 256 and dtype == torch.bfloat16:
                        continue
                    with torch.random.fork_rng(devices=rng_device):
                        with pytorch_op_timer():
                            res1 = torch.randperm(
                                n, dtype=dtype, device=device)
                    res2 = torch.empty(0, dtype=dtype, device=device)
                    with pytorch_op_timer():
                        torch.randperm(n, out=res2, dtype=dtype, device=device)
                    self.assertEqual(res1, res2, atol=0, rtol=0)
                    self.assertEqual(res1.sort().values.long(),
                                     torch.arange(n, device=device))

        # Default type is long
        for n in (100, 10000):
            with pytorch_op_timer():
                result = torch.randperm(n, device=device)
            self.assertEqual(result.dtype, torch.long)

        # randperm of 0 elements is an empty tensor
        with pytorch_op_timer():
            res1 = torch.randperm(0)
        res2 = torch.tensor(5, dtype=dtype, device=device)
        with pytorch_op_timer():
            torch.randperm(0, out=res2)
        self.assertEqual(res1.numel(), 0)
        self.assertEqual(res2.numel(), 0)

        # Test exceptions when n is too large for a floating point type
        for dtype, small_n, large_n in ((torch.uint8, 2**8, 2**8 + 1),
                                        (torch.half, 2**11 + 1, 2**11 + 2),
                                        (torch.float, 2**24 + 1, 2**24 + 2),
                                        (torch.double, 2**25,  # 2**53 + 1 is too large to run
                                         2**53 + 2)):
            res = torch.empty(0, dtype=dtype, device=device)
            with pytorch_op_timer():
                torch.randperm(small_n, out=res)  # No exception expected
            self.assertRaises(RuntimeError, lambda: torch.randperm(
                large_n, out=res, device=device))

        # Test non-contiguous tensors
        for n in (4, 5, 6, 10, 20):
            non_contiguous_tensor = torch.zeros(
                (2, 3), dtype=torch.long, device=device).t()
            # self.assertFalse(non_contiguous_tensor.is_contiguous())
            with torch.random.fork_rng(devices=rng_device):

                with pytorch_op_timer():
                    res = torch.randperm(n, dtype=torch.long, device=device)
            with pytorch_op_timer():
                torch.randperm(n, out=non_contiguous_tensor)
            self.assertEqual(non_contiguous_tensor, res)
            self.assertEqual(res.sort().values.long(),
                             torch.arange(n, device=device))

    # Test exceptions when device and generator types are incompatible
    @onlyAcceleratedDeviceTypes
    def test_randperm_device_compatibility(self, device):
        cuda_gen = torch.Generator(device=device)
        cpu_gen = torch.Generator(device='cpu')

        # n=0 is a special case that we don't need to use generator, thus no error even if
        # device and generator don't match
        with pytorch_op_timer():
            torch.randperm(0, device=device,
                           generator=torch.Generator(device=device))
        with pytorch_op_timer():
            torch.randperm(0, device=device,
                           generator=torch.Generator(device='cpu'))
        with pytorch_op_timer():
            torch.randperm(0, device='cpu',
                           generator=torch.Generator(device=device))

        for n in (1, 3, 100, 30000):
            with pytorch_op_timer():
                torch.randperm(n, device=device,
                               generator=torch.Generator(device=device))
            with pytorch_op_timer():
                torch.randperm(n, device=device,
                               generator=torch.Generator(device=device))
            # For cuda:0 to match cuda:1, we are making consistent device type matching
            # behavior just like torch.randint. Longer term, generator should ignore
            # device ordinal, since it's not used anyway.
            torch.randint(low=0, high=n + 1, size=(1,), device=device,
                          generator=torch.Generator(device=device))
            with pytorch_op_timer():
                torch.randperm(n, device=device,
                               generator=torch.Generator(device=device))

            regex = 'Expected a .* device type for generator but found .*'
            cuda_t = torch.tensor(n, device=device)
            self.assertRaisesRegex(RuntimeError, regex, lambda: torch.randperm(
                n, device=device, generator=cpu_gen))
            self.assertRaisesRegex(RuntimeError, regex, lambda: torch.randperm(
                n, device=device, generator=cpu_gen, out=cuda_t))
            cpu_t = torch.tensor(n, device='cpu')
            self.assertRaisesRegex(RuntimeError, regex, lambda: torch.randperm(
                n, device='cpu', generator=cuda_gen))
            self.assertRaisesRegex(RuntimeError, regex, lambda: torch.randperm(
                n, device='cpu', generator=cuda_gen, out=cpu_t))
            self.assertRaisesRegex(RuntimeError, regex, lambda: torch.randperm(
                n, generator=cuda_gen))  # implicitly on CPU


instantiate_device_type_tests(TestRandomTensorCreation, globals())


if __name__ == '__main__':
    run_tests()
