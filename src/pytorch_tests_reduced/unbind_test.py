# Owner(s): ["module: tests"]

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

# Tests ops and indexing to ensure they return views (and new tensors) as
# appropriate.


class TestViewOps(TestCase):

    # TODO: opinfo this or move to unbind's test suite
    def test_unbind(self, device):
        stacked = torch.randn(3, 10, 10, requires_grad=True, device=device)

        with pytorch_op_timer():
            x, y, z = stacked.unbind()
        grad = torch.randn(3, 10, 10, device=device)

        torch.autograd.backward([x, y, z], grad.unbind())
        self.assertEqual(stacked.grad, grad)
        # check that it works with only one gradient provided (#9977)
        for i in range(3):
            stacked = torch.randn(3, 10, 10, requires_grad=True, device=device)
            with pytorch_op_timer():
                outs = stacked.unbind()
            with pytorch_op_timer():
                gi = grad.unbind()[i]
            g, = torch.autograd.grad(outs[i], stacked, gi)
            g_expected = torch.stack([gi if j == i else torch.zeros_like(gi, device=device)
                                      for j in range(3)], dim=0)
            self.assertEqual(g, g_expected)
        # Check with gradcheck
        stacked = torch.randn(
            3, 10, 10, dtype=torch.double, requires_grad=True, device=device)
        gradcheck(lambda x: x.unbind(), (stacked,), check_forward_ad=True)


instantiate_device_type_tests(TestViewOps, globals())

if __name__ == '__main__':
    run_tests()
