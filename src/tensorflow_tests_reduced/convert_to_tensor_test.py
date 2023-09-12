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
from ..utils.timer_wrapper import tensorflow_op_timer

@test_util.run_all_in_graph_and_eager_modes
class IndexedSlicesTest(test_util.TensorFlowTestCase):

  def testToTensor(self):
    values = constant_op.constant([2, 3, 5, 7], shape=[2, 2])
    indices = constant_op.constant([0, 2])
    x = indexed_slices.IndexedSlices(values, indices)
    with self.assertRaises(ValueError):
      tensor = ops.convert_to_tensor(x, name="tensor")
    self.assertEqual(tensor_shape.TensorShape(None), x.shape)

    dense_shape = constant_op.constant([3, 2])
    y = indexed_slices.IndexedSlices(values, indices, dense_shape)
    timer = tensorflow_op_timer()
    with timer:
      tensor = ops.convert_to_tensor(y, name="tensor")
      timer.gen.send(tensor)
    self.assertAllEqual(tensor.shape, y.shape)
    self.assertAllEqual(self.evaluate(tensor), [[2, 3], [0, 0], [5, 7]])


class OperationTest(test_util.TensorFlowTestCase):

  @test_util.run_in_graph_and_eager_modes
  def testConvertToTensorNestedArray(self):
    values = [[2], [3], [5], [7]]
    timer = tensorflow_op_timer()
    with timer:
      tensor = ops.convert_to_tensor(values)
      timer.gen.send(tensor)
    self.assertAllEqual((4, 1), tensor.get_shape().as_list())
    self.assertAllEqual(values, self.evaluate(tensor))

  def testConvertToTensorEager(self):
    with context.eager_mode():
      t = constant_op.constant(1)
      self.assertTrue(isinstance(t, ops.EagerTensor))
      timer = tensorflow_op_timer()
      with timer:
        converted = ops.convert_to_tensor(t)
        timer.gen.send(converted)
      self.assertTrue(isinstance(converted, ops.EagerTensor))
      timer = tensorflow_op_timer()
      with timer:
        converted = ops.convert_to_tensor(1)
        timer.gen.send(converted)
      self.assertTrue(isinstance(converted, ops.EagerTensor))

  @test_util.run_in_graph_and_eager_modes
  def testConvertToTensorNestedTuple(self):
    values = ((2,), (3,), (5,), (7,))
    timer = tensorflow_op_timer()
    with timer:
      tensor = ops.convert_to_tensor(values)
      timer.gen.send(tensor)
    self.assertAllEqual((4, 1), tensor.get_shape().as_list())
    self.assertAllEqual(values, self.evaluate(ops.convert_to_tensor(values)))

  @test_util.run_in_graph_and_eager_modes
  def testConvertToTensorNestedTensors(self):
    values = ((2,), (3,), (5,), (7,))
    timer = tensorflow_op_timer()
    with timer:
      tensor = ops.convert_to_tensor(
        [constant_op.constant(row) for row in values])
      timer.gen.send(tensor)
    self.assertAllEqual((4, 1), tensor.get_shape().as_list())
    self.assertAllEqual(values, self.evaluate(tensor))
    timer = tensorflow_op_timer()
    with timer:
      tensor = ops.convert_to_tensor(
        [[constant_op.constant(v) for v in row] for row in values])
      timer.gen.send(tensor)
    self.assertAllEqual((4, 1), tensor.get_shape().as_list())
    self.assertAllEqual(values, self.evaluate(tensor))

  @test_util.run_in_graph_and_eager_modes
  def testConvertToTensorNestedMix(self):
    values = ([2], (3,), [constant_op.constant(5)], constant_op.constant([7]))
    timer = tensorflow_op_timer()
    with timer:
      tensor = ops.convert_to_tensor(values)
      timer.gen.send(tensor)
    self.assertAllEqual((4, 1), tensor.get_shape().as_list())
    self.assertAllEqual(((2,), (3,), (5,), (7,)), self.evaluate(tensor))

  @test_util.run_in_graph_and_eager_modes
  def testConvertToTensorPreferred(self):
    values = [2, 3, 5, 7]
    tensor = ops.convert_to_tensor(values, preferred_dtype=dtypes.float32)
    self.assertEqual(dtypes.float32, tensor.dtype)

    # Convert empty tensor to anything.
    values = []
    timer = tensorflow_op_timer()
    with timer:
      tensor = ops.convert_to_tensor(values, preferred_dtype=dtypes.int64)
      timer.gen.send(tensor)
    self.assertEqual(dtypes.int64, tensor.dtype)

    # The preferred dtype is a type error and will convert to
    # float32 instead.
    values = [1.23]
    timer = tensorflow_op_timer()
    with timer:
      tensor = ops.convert_to_tensor(values, preferred_dtype=dtypes.int64)
      timer.gen.send(tensor)
    self.assertEqual(dtypes.float32, tensor.dtype)

  @test_util.run_in_graph_and_eager_modes
  def testConvertToInvalidTensorType(self):
    with self.assertRaises(TypeError):
      # Forcing an invalid dtype should fail with a type error.
      values = [1.23]
      timer = tensorflow_op_timer()
      with timer:
        ops.convert_to_tensor(values, dtype=dtypes.int64)
        timer.gen.send(ops)

  @test_util.run_in_graph_and_eager_modes
  def testConvertToLongLongTensorType(self):
    timer = tensorflow_op_timer()
    with timer:
      tensor = ops.convert_to_tensor(
        # Get a numpy array of dtype NPY_LONGLONG
        np.prod(constant_op.constant([1])._shape_tuple()),
        dtype=dtypes.int64)
      timer.gen.send(tensor)
    self.assertEqual(dtypes.int64, tensor.dtype)

  @test_util.run_in_graph_and_eager_modes
  def testConvertToTensorFromInvalidTensor(self):
    tensor = constant_op.constant(42.0, dtype=dtypes.float32)
    with self.assertRaises(ValueError):
      ops.convert_to_tensor(tensor, dtype=dtypes.int32)

  @test_util.run_in_graph_and_eager_modes
  def testConvertToTensorProtocol(self):
    class TensorCompatible:

      def __tf_tensor__(self, dtype=None, name=None):
        return constant_op.constant((1, 2, 3), dtype=dtype, name=name)

    tc = TensorCompatible()
    timer = tensorflow_op_timer()
    with timer:
      tensor = ops.convert_to_tensor(tc, dtype=dtypes.int32)
      timer.gen.send(tensor)
    self.assertEqual(tensor.dtype, dtypes.int32)
    self.assertAllEqual((1, 2, 3), self.evaluate(tensor))

  @test_util.run_deprecated_v1
  def testNoConvert(self):
    # Operation cannot be converted to Tensor.
    op = control_flow_ops.no_op()
    with self.assertRaisesRegex(TypeError,
                                "can't convert Operation '.+' to Tensor"):
      ops.convert_to_tensor(op)

if __name__ == "__main__":
  googletest.main()
