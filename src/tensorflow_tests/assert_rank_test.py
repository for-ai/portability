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


class AssertRankTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_rank_zero_tensor_raises_if_rank_too_small_static_rank(self):
    tensor = constant_op.constant(1, name="my_tensor")
    desired_rank = 1
    with self.assertRaisesRegex(ValueError, "fail.*must have rank 1"):
      with ops.control_dependencies(
          [check_ops.assert_rank(
              tensor, desired_rank, message="fail")]):
        self.evaluate(array_ops.identity(tensor))

  @test_util.run_deprecated_v1
  def test_rank_zero_tensor_raises_if_rank_too_small_dynamic_rank(self):
    with self.cached_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 1
      with ops.control_dependencies(
          [check_ops.assert_rank(
              tensor, desired_rank, message="fail")]):
        with self.assertRaisesOpError("fail.*my_tensor.*rank"):
          array_ops.identity(tensor).eval(feed_dict={tensor: 0})

  @test_util.run_in_graph_and_eager_modes
  def test_rank_zero_tensor_doesnt_raise_if_rank_just_right_static_rank(self):
    tensor = constant_op.constant(1, name="my_tensor")
    desired_rank = 0
    with ops.control_dependencies(
        [check_ops.assert_rank(tensor, desired_rank)]):
      self.evaluate(array_ops.identity(tensor))

  @test_util.run_deprecated_v1
  def test_rank_zero_tensor_doesnt_raise_if_rank_just_right_dynamic_rank(self):
    with self.cached_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 0
      with ops.control_dependencies(
          [check_ops.assert_rank(tensor, desired_rank)]):
        array_ops.identity(tensor).eval(feed_dict={tensor: 0})

  @test_util.run_in_graph_and_eager_modes
  def test_rank_one_tensor_raises_if_rank_too_large_static_rank(self):
    tensor = constant_op.constant([1, 2], name="my_tensor")
    desired_rank = 0
    with self.assertRaisesRegex(ValueError, "rank"):
      with ops.control_dependencies(
          [check_ops.assert_rank(tensor, desired_rank)]):
        self.evaluate(array_ops.identity(tensor))

  @test_util.run_deprecated_v1
  def test_rank_one_tensor_raises_if_rank_too_large_dynamic_rank(self):
    with self.cached_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 0
      with ops.control_dependencies(
          [check_ops.assert_rank(tensor, desired_rank)]):
        with self.assertRaisesOpError("my_tensor.*rank"):
          array_ops.identity(tensor).eval(feed_dict={tensor: [1, 2]})

  @test_util.run_in_graph_and_eager_modes
  def test_rank_one_tensor_doesnt_raise_if_rank_just_right_static_rank(self):
    tensor = constant_op.constant([1, 2], name="my_tensor")
    desired_rank = 1
    with ops.control_dependencies(
        [check_ops.assert_rank(tensor, desired_rank)]):
      self.evaluate(array_ops.identity(tensor))

  @test_util.run_deprecated_v1
  def test_rank_one_tensor_doesnt_raise_if_rank_just_right_dynamic_rank(self):
    with self.cached_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 1
      with ops.control_dependencies(
          [check_ops.assert_rank(tensor, desired_rank)]):
        array_ops.identity(tensor).eval(feed_dict={tensor: [1, 2]})

  @test_util.run_in_graph_and_eager_modes
  def test_rank_one_tensor_raises_if_rank_too_small_static_rank(self):
    tensor = constant_op.constant([1, 2], name="my_tensor")
    desired_rank = 2
    with self.assertRaisesRegex(ValueError, "rank"):
      with ops.control_dependencies(
          [check_ops.assert_rank(tensor, desired_rank)]):
        self.evaluate(array_ops.identity(tensor))

  @test_util.run_deprecated_v1
  def test_rank_one_tensor_raises_if_rank_too_small_dynamic_rank(self):
    with self.cached_session():
      tensor = array_ops.placeholder(dtypes.float32, name="my_tensor")
      desired_rank = 2
      with ops.control_dependencies(
          [check_ops.assert_rank(tensor, desired_rank)]):
        with self.assertRaisesOpError("my_tensor.*rank"):
          array_ops.identity(tensor).eval(feed_dict={tensor: [1, 2]})

  @test_util.run_in_graph_and_eager_modes
  def test_raises_if_rank_is_not_scalar_static(self):
    tensor = constant_op.constant([1, 2], name="my_tensor")
    with self.assertRaisesRegex(ValueError, "Rank must be a scalar"):
      check_ops.assert_rank(tensor, np.array([], dtype=np.int32))

  @test_util.run_deprecated_v1
  def test_raises_if_rank_is_not_scalar_dynamic(self):
    with self.cached_session():
      tensor = constant_op.constant(
          [1, 2], dtype=dtypes.float32, name="my_tensor")
      rank_tensor = array_ops.placeholder(dtypes.int32, name="rank_tensor")
      with self.assertRaisesOpError("Rank must be a scalar"):
        with ops.control_dependencies(
            [check_ops.assert_rank(tensor, rank_tensor)]):
          array_ops.identity(tensor).eval(feed_dict={rank_tensor: [1, 2]})

  @test_util.run_in_graph_and_eager_modes
  def test_raises_if_rank_is_not_integer_static(self):
    tensor = constant_op.constant([1, 2], name="my_tensor")
    with self.assertRaisesRegex(TypeError, "must be of type tf.int32"):
      check_ops.assert_rank(tensor, .5)

  @test_util.run_deprecated_v1
  def test_raises_if_rank_is_not_integer_dynamic(self):
    with self.cached_session():
      tensor = constant_op.constant(
          [1, 2], dtype=dtypes.float32, name="my_tensor")
      rank_tensor = array_ops.placeholder(dtypes.float32, name="rank_tensor")
      with self.assertRaisesRegex(TypeError, "must be of type tf.int32"):
        with ops.control_dependencies(
            [check_ops.assert_rank(tensor, rank_tensor)]):
          array_ops.identity(tensor).eval(feed_dict={rank_tensor: .5})

if __name__ == "__main__":
  test.main()
