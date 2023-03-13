# Owner(s): ["module: tests"]

import torch
import numpy as np

import math
from typing import Dict, List, Sequence
import random
from functools import partial
from itertools import product, combinations, permutations
import warnings

from torch._six import inf, nan
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import (
    all_types_and_complex_and, get_all_math_dtypes, integral_types, complex_types, floating_types_and,
    integral_types_and, floating_and_complex_types_and, all_types_and,
)
from torch.testing._internal.common_utils import (
    TestCase, run_tests, skipIfNoSciPy, slowTest, torch_to_numpy_dtype_dict,
    IS_WINDOWS)
from torch.testing._internal.common_device_type import (
    OpDTypes, expectedFailureMeta, instantiate_device_type_tests, onlyCPU, dtypes, dtypesIfCUDA, dtypesIfCPU,
    onlyNativeDeviceTypes, onlyCUDA, largeTensorTest, ops, precisionOverride)
from torch.testing._internal.common_methods_invocations import (
    ReductionOpInfo, reduction_ops, reference_masked_ops)

from ..utils.pytorch_device_decorators import onlyNativeDeviceTypes, onlyAcceleratedDeviceTypes, instantiate_device_type_tests
from ..utils.timer_wrapper import pytorch_op_timer


class TestSum(TestCase):
    def test_sum(self, device):
        t = torch.tensor([1, 2, 3], device=device)
        self.assertEqual(torch.sum(t), torch.tensor(6))
    
instantiate_device_type_tests(TestSum, globals())

if __name__ == '__main__':
    run_tests()
