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
from ..utils.pytorch_device_decorators import onlyNativeDeviceTypes, onlyAcceleratedDeviceTypes, instantiate_device_type_tests
from ..utils.timer_wrapper import pytorch_op_timer

class TestTensorCreation(TestCase):
    exact_dtype = True

    # TODO: this test should be updated
    @suppress_warnings
    def test_range(self, device):
        torch.set_default_dtype(torch.float32)
        with pytorch_op_timer():
            res1 = torch.range(0, 1, device=device)
        res2 = torch.tensor((), device=device)
        with pytorch_op_timer():
            torch.range(0, 1, device=device, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)

        # Check range for non-contiguous tensors.
        x = torch.zeros(2, 3, device=device)
        with pytorch_op_timer():
            torch.range(0, 3, device=device, out=x.narrow(1, 1, 2))
        res2 = torch.tensor(((0, 0, 1), (0, 2, 3)),
                            device=device, dtype=torch.float32)
        self.assertEqual(x, res2, atol=1e-16, rtol=0)

        # Check negative
        res1 = torch.tensor((1, 0), device=device, dtype=torch.float32)
        res2 = torch.tensor((), device=device)
        with pytorch_op_timer():
            torch.range(1, 0, -1, device=device, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)

        # Equal bounds
        res1 = torch.ones(1, device=device)
        res2 = torch.tensor((), device=device)
        with pytorch_op_timer():
            torch.range(1, 1, -1, device=device, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)
        with pytorch_op_timer():
            torch.range(1, 1, 1, device=device, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)

    # TODO: this test should be updated
    def test_range_warning(self, device):
        with warnings.catch_warnings(record=True) as w:
            with pytorch_op_timer():
                torch.range(0, 10, device=device)
            self.assertEqual(len(w), 1)


instantiate_device_type_tests(TestTensorCreation, globals())

if __name__ == '__main__':
    run_tests()
