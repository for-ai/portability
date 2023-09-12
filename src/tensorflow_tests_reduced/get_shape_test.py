# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.python.framework.ops."""

import gc
import os
import threading
import weakref

from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import full_type_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as eager_function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import versions
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import resources
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
import tensorflow.python.ops.gradients  # pylint: disable=unused-import
from tensorflow.python.platform import googletest
from tensorflow.python.util import compat
# import tensorflow as tf
from ..utils.timer_wrapper import tensorflow_op_timer


class TensorAndShapeTest(test_util.TensorFlowTestCase):
  def testShape(self):
    op = ops.Operation(
        ops._NodeDef("FloatOutput", "myop"), ops.Graph(), [], [dtypes.float32])
    t = op.outputs[0]
    self.assertEqual(tensor_shape.unknown_shape(), t.get_shape())
    timer = tensorflow_op_timer()
    with timer:
      t.set_shape([1, 2, 3])
      timer.gen.send(None)
    self.assertEqual([1, 2, 3], t.get_shape())

  def testAddShape(self):
    with self.cached_session():
      a = array_ops.zeros([2, 3])
      b = array_ops.ones([1, 3])
      c = a + b
      timer = tensorflow_op_timer()
      with timer:
        c.shape
        timer.gen.send(None)
      self.assertEqual([2, 3], c.shape)

  @test_util.run_deprecated_v1
  def testUnknownDim(self):
    with self.cached_session():
      a = array_ops.placeholder(dtype=dtypes.float32, shape=[2, None, 3])
      b = array_ops.placeholder(dtype=dtypes.float32, shape=[2, None, 3])
      c = a + b
      timer = tensorflow_op_timer()
      with timer:
        c.shape
        timer.gen.send(None)
      self.assertEqual([2, None, 3], c.shape.as_list())

  @test_util.run_deprecated_v1
  def testUnknownShape(self):
    with self.cached_session():
      a = array_ops.placeholder(dtype=dtypes.float32, shape=None)
      b = array_ops.ones([1, 3])
      c = a + b
      timer = tensorflow_op_timer()
      with timer:
        c.shape
        timer.gen.send(None)
      self.assertEqual(tensor_shape.unknown_shape(), c.shape)

  @test_util.run_deprecated_v1
  def testScalarShape(self):
    with self.cached_session():
      a = array_ops.placeholder(dtype=dtypes.float32, shape=[])
      b = array_ops.ones([])
      c = a + b
      timer = tensorflow_op_timer()
      with timer:
        c.shape
        timer.gen.send(None)
      self.assertEqual(tensor_shape.TensorShape([]), c.shape)

  
  
if __name__ == "__main__":
  googletest.main()
