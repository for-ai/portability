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
"""Tests for tf numpy array methods."""

import itertools
import operator
import sys
from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_math_ops
from tensorflow.python.platform import test


_virtual_devices_ready = False


def set_up_virtual_devices():
  global _virtual_devices_ready
  if _virtual_devices_ready:
    return
  physical_devices = config.list_physical_devices('CPU')
  config.set_logical_device_configuration(
      physical_devices[0], [
          context.LogicalDeviceConfiguration(),
          context.LogicalDeviceConfiguration()
      ])
  _virtual_devices_ready = True



  def testArray(self):
    ndmins = [0, 1, 2, 5]
    for a, dtype, ndmin, copy in itertools.product(self.all_arrays,
                                                   self.all_types, ndmins,
                                                   [True, False]):
      self.match(
          np_array_ops.array(a, dtype=dtype, ndmin=ndmin, copy=copy),
          np.array(a, dtype=dtype, ndmin=ndmin, copy=copy))

    zeros_list = np_array_ops.zeros(5)

    def test_copy_equal_false():
      # Backing tensor is the same if copy=False, other attributes being None.
      self.assertIs(np_array_ops.array(zeros_list, copy=False), zeros_list)
      self.assertIs(np_array_ops.array(zeros_list, copy=False), zeros_list)

      # Backing tensor is different if ndmin is not satisfied.
      self.assertIsNot(
          np_array_ops.array(zeros_list, copy=False, ndmin=2),
          zeros_list)
      self.assertIsNot(
          np_array_ops.array(zeros_list, copy=False, ndmin=2),
          zeros_list)
      self.assertIs(
          np_array_ops.array(zeros_list, copy=False, ndmin=1),
          zeros_list)
      self.assertIs(
          np_array_ops.array(zeros_list, copy=False, ndmin=1),
          zeros_list)

      # Backing tensor is different if dtype is not satisfied.
      self.assertIsNot(
          np_array_ops.array(zeros_list, copy=False, dtype=int),
          zeros_list)
      self.assertIsNot(
          np_array_ops.array(zeros_list, copy=False, dtype=int),
          zeros_list)
      self.assertIs(
          np_array_ops.array(zeros_list, copy=False, dtype=float),
          zeros_list)
      self.assertIs(
          np_array_ops.array(zeros_list, copy=False, dtype=float),
          zeros_list)

    test_copy_equal_false()
    with ops.device('CPU:1'):
      test_copy_equal_false()

    self.assertNotIn('CPU:1', zeros_list.backing_device)
    with ops.device('CPU:1'):
      self.assertIn(
          'CPU:1', np_array_ops.array(zeros_list, copy=True).backing_device)
      self.assertIn(
          'CPU:1', np_array_ops.array(np.array(0), copy=True).backing_device)

  
 

if __name__ == '__main__':
  ops.enable_eager_execution()
  ops.enable_numpy_style_type_promotion()
  np_math_ops.enable_numpy_methods_on_tensor()
  test.main()