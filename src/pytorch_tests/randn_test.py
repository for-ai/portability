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

    @dtypes(torch.half, torch.float, torch.bfloat16, torch.double,
            torch.complex32, torch.complex64, torch.complex128)
    def test_randn(self, device, dtype):
        SIZE = 100
        for size in [0, SIZE]:
            torch.manual_seed(123456)
            res1 = torch.randn(size, size, dtype=dtype, device=device)
            res2 = torch.tensor([], dtype=dtype, device=device)
            torch.manual_seed(123456)
            torch.randn(size, size, out=res2)
            self.assertEqual(res1, res2)


instantiate_device_type_tests(TestTensorCreation, globals())

if __name__ == '__main__':
    run_tests()
