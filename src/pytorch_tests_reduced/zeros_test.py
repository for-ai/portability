
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

from torch.utils.dlpack import to_dlpack

from ..utils.pytorch_device_decorators import onlyNativeDeviceTypes, onlyAcceleratedDeviceTypes, instantiate_device_type_tests, onlyGPU, onlyTPU, skipCUDAIfNoCudnn
from ..utils.timer_wrapper import pytorch_op_timer

# TODO: replace with make_tensor
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
            with pytorch_op_timer():
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

class TestTensorCreation(TestCase):
    exact_dtype = True


    # FIXME: Create an OpInfo-based tensor creation method test that verifies this for all tensor
    #   creation methods and verify all dtypes and layouts
    @dtypes(torch.bool, torch.uint8, torch.int16, torch.int64, torch.float16, torch.float32, torch.complex64)
    def test_zeros_dtype_layout_device_match(self, device, dtype):
        layout = torch.strided
        with pytorch_op_timer():
            t = torch.zeros((2, 3), device=device, dtype=dtype, layout=layout)
        self.assertIs(dtype, t.dtype)
        self.assertIs(layout, t.layout)
        self.assertEqual(torch.device(device), t.device)

    # TODO: this test should be updated
    def test_zeros(self, device):
        with pytorch_op_timer():
            res1 = torch.zeros(100, 100, device=device)
        with pytorch_op_timer():
            res2 = torch.tensor((), device=device)
        with pytorch_op_timer():
            torch.zeros(100, 100, device=device, out=res2)

        self.assertEqual(res1, res2)

        with pytorch_op_timer():
            boolTensor = torch.zeros(2, 2, device=device, dtype=torch.bool)
        expected = torch.tensor([[False, False], [False, False]],
                                device=device, dtype=torch.bool)
        self.assertEqual(boolTensor, expected)

        with pytorch_op_timer():
              halfTensor = torch.zeros(1, 1, device=device, dtype=torch.half)
        expected = torch.tensor([[0.]], device=device, dtype=torch.float16)
        self.assertEqual(halfTensor, expected)

        with pytorch_op_timer():
              bfloat16Tensor = torch.zeros(1, 1, device=device, dtype=torch.bfloat16)
        expected = torch.tensor([[0.]], device=device, dtype=torch.bfloat16)
        self.assertEqual(bfloat16Tensor, expected)

        with pytorch_op_timer():
            complexTensor = torch.zeros(2, 2, device=device, dtype=torch.complex64)
        expected = torch.tensor([[0., 0.], [0., 0.]], device=device, dtype=torch.complex64)
        self.assertEqual(complexTensor, expected)

        with pytorch_op_timer():
            complexHalfTensor = torch.zeros(2, 2, device=device, dtype=torch.complex32)
        expected = torch.tensor([[0., 0.], [0., 0.]], device=device, dtype=torch.complex32)
        self.assertEqual(complexHalfTensor, expected)

    # TODO: this test should be updated
    def test_zeros_out(self, device):
        shape = (3, 4)
        with pytorch_op_timer():
            out = torch.zeros(shape, device=device)
        with pytorch_op_timer():
            torch.zeros(shape, device=device, out=out)

        # change the dtype, layout, device
        with self.assertRaises(RuntimeError):
            torch.zeros(shape, device=device, dtype=torch.int64, out=out)
        with self.assertRaises(RuntimeError):
            torch.zeros(shape, device=device, layout=torch.sparse_coo, out=out)


        with pytorch_op_timer():
            first = torch.zeros(shape, device=device)
        with pytorch_op_timer():
            second = torch.zeros(shape, device=device, dtype=out.dtype, out=out)
        # leave them the same

        
        self.assertEqual(first, second)

        with pytorch_op_timer():
            first = torch.zeros(shape, device=device)
        with pytorch_op_timer():
            second = torch.zeros(shape, device=device, layout=torch.strided, out=out)

        self.assertEqual(first, second)

        with pytorch_op_timer():
            first = torch.zeros(shape, device=device)
        with pytorch_op_timer():
            second = torch.zeros(shape, device=device, out=out)

        self.assertEqual(first, second)
    
# Class for testing *like ops, like torch.ones_like
class TestLikeTensorCreation(TestCase):
    exact_dtype = True

    def test_zeros_like(self, device):
        with pytorch_op_timer():
            expected = torch.zeros((100, 100,), device=device)

        res1 = torch.zeros_like(expected)
        self.assertEqual(res1, expected)

    @deviceCountAtLeast(2)
    def test_zeros_like_multiple_device(self, devices):
        with pytorch_op_timer():
            expected = torch.zeros(100, 100, device=devices[0])
        x = torch.randn(100, 100, device=devices[1], dtype=torch.float32)
        output = torch.zeros_like(x)
        self.assertEqual(output, expected)

    
instantiate_device_type_tests(TestTensorCreation, globals())
instantiate_device_type_tests(TestLikeTensorCreation, globals())


if __name__ == '__main__':
    run_tests()
