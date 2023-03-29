# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.check_ops."""

import time

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test
from ..utils.timer_wrapper import tensorflow_op_timer


class AssertEqualTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_noop_when_both_identical(self):
    larry = constant_op.constant([])
    check_op = check_ops.assert_equal(larry, larry)
    if context.executing_eagerly():
      self.assertIs(check_op, None)
    else:
      with tensorflow_op_timer():
        self.assertEqual(check_op.type, "NoOp")


class EnsureShapeTest(test.TestCase):

  # Static shape inference
  @test_util.run_deprecated_v1
  def testStaticShape(self):
    placeholder = array_ops.placeholder(dtypes.int32)
    ensure_shape_op = check_ops.ensure_shape(placeholder, (3, 3, 3))
    with tensorflow_op_timer():
      self.assertEqual(ensure_shape_op.get_shape(), (3, 3, 3))

  @test_util.run_deprecated_v1
  def testStaticShape_MergesShapes(self):
    placeholder = array_ops.placeholder(dtypes.int32, shape=(None, None, 3))
    ensure_shape_op = check_ops.ensure_shape(placeholder, (5, 4, None))
    with tensorflow_op_timer():
      self.assertEqual(ensure_shape_op.get_shape(), (5, 4, 3))

  @test_util.run_deprecated_v1
  def testStaticShape_RaisesErrorWhenRankIncompatible(self):
    placeholder = array_ops.placeholder(dtypes.int32, shape=(None, None, 3))
    with self.assertRaises(ValueError):
      check_ops.ensure_shape(placeholder, (2, 3))

  @test_util.run_deprecated_v1
  def testStaticShape_RaisesErrorWhenDimIncompatible(self):
    placeholder = array_ops.placeholder(dtypes.int32, shape=(None, None, 3))
    with self.assertRaises(ValueError):
      check_ops.ensure_shape(placeholder, (2, 2, 4))

  @test_util.run_deprecated_v1
  def testStaticShape_CanSetUnknownShape(self):
    placeholder = array_ops.placeholder(dtypes.int32)
    derived = placeholder / 3
    ensure_shape_op = check_ops.ensure_shape(derived, None)
    with tensorflow_op_timer():
      self.assertEqual(ensure_shape_op.get_shape(), None)

if __name__ == "__main__":
  test.main()
