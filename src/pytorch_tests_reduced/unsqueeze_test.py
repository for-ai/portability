
import torch
import numpy as np

import unittest
from itertools import product, permutations, combinations
from functools import partial
import random

from torch.testing import make_tensor
from torch.testing._internal.common_utils import (
    TestCase, run_tests, suppress_warnings, gradcheck, gradgradcheck,
    numpy_to_torch_dtype_dict,
)
from torch.testing._internal.common_device_type import \
    (onlyCPU,
     dtypes, onlyNativeDeviceTypes, skipMeta)
from torch.testing._internal.common_dtype import (
    all_types_and_complex_and, complex_types, all_types_and, floating_and_complex_types_and,
)


from ..utils.pytorch_device_decorators import onlyNativeDeviceTypes, onlyAcceleratedDeviceTypes, instantiate_device_type_tests
from ..utils.timer_wrapper import pytorch_op_timer

class TestViewOps(TestCase):
    def is_view_of(self, base, other):
        if (not other._is_view() or
            other is base or
            other._base is not base or
                base.device != other.device):
            return False
        # Note: only validates storage on native device types
        # because some accelerators, like XLA, do not expose storage
        if base.device.type == 'cpu' or base.device.type == 'cuda':
            if base.storage().data_ptr() != other.storage().data_ptr():
                return False

        return True

    def test_unsqueeze_view(self, device):
        t = torch.ones(5, 5, device=device)
        with pytorch_op_timer():
            v = torch.unsqueeze(t, 1)
        self.assertTrue(self.is_view_of(t, v))

        v[0, 0, 1] = 0
        self.assertEqual(t[0, 1], v[0, 0, 1])


instantiate_device_type_tests(TestViewOps, globals())

if __name__ == '__main__':
    run_tests()
