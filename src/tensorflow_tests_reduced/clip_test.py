# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for tf numpy mathematical methods."""

import itertools
from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_math_ops
from tensorflow.python.platform import test
from tensorflow.python.ops.numpy_ops import np_config
from ..utils.timer_wrapper import tensorflow_op_timer

class MathTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(MathTest, self).setUp()
    self.array_transforms = [
        lambda x: x,  # Identity,
        ops.convert_to_tensor,
        np.array,
        lambda x: np.array(x, dtype=np.float32),
        lambda x: np.array(x, dtype=np.float64),
        np_array_ops.array,
        lambda x: np_array_ops.array(x, dtype=np.float32),
        lambda x: np_array_ops.array(x, dtype=np.float64),
    ]
    self.types = [np.int32, np.int64, np.float32, np.float64]
  def match(self, actual, expected, msg='', check_dtype=True):
    self.assertIsInstance(actual, np_arrays.ndarray)
    if check_dtype:
      self.assertEqual(
          actual.dtype, expected.dtype,
          'Dtype mismatch.\nActual: {}\nExpected: {}\n{}'.format(
              actual.dtype.as_numpy_dtype, expected.dtype, msg))
    self.assertEqual(
        actual.shape, expected.shape,
        'Shape mismatch.\nActual: {}\nExpected: {}\n{}'.format(
            actual.shape, expected.shape, msg))
    np.testing.assert_allclose(actual.tolist(), expected.tolist(), rtol=1e-6)

  def _testBinaryOp(self,
                    math_fun,
                    np_fun,
                    name,
                    operands=None,
                    extra_operands=None,
                    check_promotion=True,
                    check_promotion_result_type=True):

    def run_test(a, b):
      for fn in self.array_transforms:
        arg1 = fn(a)
        arg2 = fn(b)
        self.match(
            math_fun(arg1, arg2),
            np_fun(arg1, arg2),
            msg='{}({}, {})'.format(name, arg1, arg2))
      # Tests type promotion
      for type_a in self.types:
        for type_b in self.types:
          if not check_promotion and type_a != type_b:
            continue
          arg1 = np_array_ops.array(a, dtype=type_a)
          arg2 = np_array_ops.array(b, dtype=type_b)
          self.match(
              math_fun(arg1, arg2),
              np_fun(arg1, arg2),
              msg='{}({}, {})'.format(name, arg1, arg2),
              check_dtype=check_promotion_result_type)

    if operands is None:
      operands = [(5, 2), (5, [2, 3]), (5, [[2, 3], [6, 7]]), ([1, 2, 3], 7),
                  ([1, 2, 3], [5, 6, 7])]
    for operand1, operand2 in operands:
      run_test(operand1, operand2)
    if extra_operands is not None:
      for operand1, operand2 in extra_operands:
        run_test(operand1, operand2)

  def testClip(self):

    def run_test(arr, *args, **kwargs):
      check_dtype = kwargs.pop('check_dtype', True)
      for fn in self.array_transforms:
        arr = fn(arr)
        with tensorflow_op_timer():
          test = np_math_ops.clip(arr, *args, **kwargs)
        self.match(
            np_math_ops.clip(arr, *args, **kwargs),
            np.clip(arr, *args, **kwargs),
            check_dtype=check_dtype)

    # NumPy exhibits weird typing behavior when a/a_min/a_max are scalars v/s
    # lists, e.g.,
    #
    # np.clip(np.array(0, dtype=np.int32), -5, 5).dtype == np.int64
    # np.clip(np.array([0], dtype=np.int32), -5, 5).dtype == np.int32
    # np.clip(np.array([0], dtype=np.int32), [-5], [5]).dtype == np.int64
    #
    # So we skip matching type. In tf-numpy the type of the output array is
    # always the same as the input array.
    run_test(0, -1, 5, check_dtype=False)
    run_test(-1, -1, 5, check_dtype=False)
    run_test(5, -1, 5, check_dtype=False)
    run_test(-10, -1, 5, check_dtype=False)
    run_test(10, -1, 5, check_dtype=False)
    run_test(10, None, 5, check_dtype=False)
    run_test(10, -1, None, check_dtype=False)
    run_test([0, 20, -5, 4], -1, 5, check_dtype=False)
    run_test([0, 20, -5, 4], None, 5, check_dtype=False)
    run_test([0, 20, -5, 4], -1, None, check_dtype=False)
    run_test([0.5, 20.2, -5.7, 4.4], -1.5, 5.1, check_dtype=False)

    run_test([0, 20, -5, 4], [-5, 0, -5, 0], [0, 5, 0, 5], check_dtype=False)
    run_test([[1, 2, 3], [4, 5, 6]], [2, 0, 2], 5, check_dtype=False)
    run_test([[1, 2, 3], [4, 5, 6]], 0, [5, 3, 1], check_dtype=False)

np_config.enable_numpy_behavior()
if __name__ == '__main__':
  ops.enable_eager_execution()
  ops.enable_numpy_style_type_promotion()
  np_math_ops.enable_numpy_methods_on_tensor()
  test.main()
