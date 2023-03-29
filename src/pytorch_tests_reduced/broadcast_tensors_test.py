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
from ..utils.pytorch_device_decorators import onlyNativeDeviceTypes, onlyAcceleratedDeviceTypes, instantiate_device_type_tests

from ..utils.timer_wrapper import pytorch_op_timer


class TestOldViewOps(TestCase):

    # @onlyCPU
    @dtypes(torch.float)
    def test_broadcast_tensors(self, device, dtype):
        x0 = torch.randn(2, 1, 3, dtype=dtype, device=device)
        x1 = torch.randn(3, dtype=dtype, device=device)
        x2 = torch.randn(3, 1, dtype=dtype, device=device)
        expected_size = (2, 3, 3)
        with pytorch_op_timer():
            y0, y1, y2 = torch.broadcast_tensors(x0, x1, x2)
        self.assertTrue(y0.size() == expected_size)
        self.assertTrue(y1.size() == expected_size)
        self.assertTrue(y2.size() == expected_size)


    # @onlyCPU
    def test_broadcast_shapes(self, device):
        examples = [(), (1,), (2,), (1, 1), (3, 1), (3, 2), (4, 1, 1), (4, 3, 2)]
        for s0 in examples:
            x0 = torch.randn(s0).to(device)
            with pytorch_op_timer():
                expected = torch.broadcast_tensors(x0)[0].shape
            actual = torch.broadcast_shapes(s0)
            self.assertEqual(expected, actual)

            for s1 in examples:
                x1 = torch.randn(s1)
                with pytorch_op_timer():
                    expected = torch.broadcast_tensors(x0, x1)[0].shape
                actual = torch.broadcast_shapes(s0, s1)
                self.assertEqual(expected, actual)

        inputs_list = [[1, 4], [4, 1], [1, 1, 3]]
        for integral_inputs in inputs_list:
            res1 = torch.broadcast_shapes(*integral_inputs)
            with pytorch_op_timer():
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
            with pytorch_op_timer():
                res2 = torch.broadcast_tensors(*map(torch.empty, s0))[0].shape
            self.assertEqual(res1, res2)

instantiate_device_type_tests(TestOldViewOps, globals())

if __name__ == '__main__':
    run_tests()
