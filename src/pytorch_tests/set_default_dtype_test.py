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
    onlyCUDA, skipCPUIf, dtypesIfCUDA, skipMeta, get_all_device_types)
from torch.testing._internal.common_dtype import (
    all_types_and_complex_and, all_types_and, floating_and_complex_types,
    floating_types, floating_and_complex_types_and, integral_types_and, get_all_dtypes
)
from torch.testing._creation import float_to_corresponding_complex_type_map

from torch.utils.dlpack import to_dlpack

# TODO: refactor tri_tests_args, _compare_trilu_indices, run_additional_tri_tests
from torch.testing._internal.common_methods_invocations import (
    tri_tests_args, _compare_trilu_indices, run_additional_tri_tests)


class TestTensorCreation(TestCase):
    # TODO: this test should be updated
    @onlyCPU
    def test_constructor_dtypes(self, device):
        default_type = torch.tensor([]).type()
        self.assertIs(torch.tensor([]).dtype, torch.get_default_dtype())

        self.assertIs(torch.uint8, torch.ByteTensor.dtype)
        self.assertIs(torch.float32, torch.FloatTensor.dtype)
        self.assertIs(torch.float64, torch.DoubleTensor.dtype)

        torch.set_default_tensor_type('torch.FloatTensor')
        self.assertIs(torch.float32, torch.get_default_dtype())
        self.assertIs(torch.FloatStorage, torch.Storage)

        # only floating-point types are supported as the default type
        self.assertRaises(
            TypeError, lambda: torch.set_default_tensor_type('torch.IntTensor'))

        torch.set_default_dtype(torch.float64)
        self.assertIs(torch.float64, torch.get_default_dtype())
        self.assertIs(torch.DoubleStorage, torch.Storage)

        torch.set_default_tensor_type(torch.FloatTensor)
        self.assertIs(torch.float32, torch.get_default_dtype())
        self.assertIs(torch.FloatStorage, torch.Storage)

        if torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            self.assertIs(torch.float32, torch.get_default_dtype())
            self.assertIs(torch.float32, torch.cuda.FloatTensor.dtype)
            self.assertIs(torch.cuda.FloatStorage, torch.Storage)

            torch.set_default_dtype(torch.float64)
            self.assertIs(torch.float64, torch.get_default_dtype())
            self.assertIs(torch.cuda.DoubleStorage, torch.Storage)

        # don't allow passing dtype to set_default_tensor_type
        self.assertRaises(
            TypeError, lambda: torch.set_default_tensor_type(torch.float32))

        # don't allow passing dtype to set_default_dtype
        for t in all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.qint8):
            # only floating-point types are supported as the default type
            if t in (
                    torch.half,
                    torch.float,
                    torch.double,
                    torch.bfloat16):
                torch.set_default_dtype(t)
            else:
                self.assertRaises(
                    TypeError, lambda: torch.set_default_dtype(t))

        torch.set_default_tensor_type(default_type)


instantiate_device_type_tests(TestTensorCreation, globals())

if __name__ == '__main__':
    run_tests()
