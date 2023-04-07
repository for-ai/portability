
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


class AssertLessTest(test.TestCase):
    
  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def test_raises_when_equal(self):
    small = constant_op.constant([1, 2], name="small")
    with self.assertRaisesOpError(  # pylint:disable=g-error-prone-assert-raises
        "failure message.*\n*.* x < y did not hold"):
      with ops.control_dependencies(
          [check_ops.assert_less(
              small, small, message="failure message")]):
        out = array_ops.identity(small)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def test_raises_when_greater(self):
    small = constant_op.constant([1, 2], name="small")
    big = constant_op.constant([3, 4], name="big")
    with self.assertRaisesOpError(  # pylint:disable=g-error-prone-assert-raises
        "x < y did not hold"):
      with ops.control_dependencies([check_ops.assert_less(big, small)]):
        out = array_ops.identity(small)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_less(self):
    small = constant_op.constant([3, 1], name="small")
    big = constant_op.constant([4, 2], name="big")
    with tensorflow_op_timer():
      test = [check_ops.assert_less(small, big)]
    with ops.control_dependencies([check_ops.assert_less(small, big)]):
      out = array_ops.identity(small)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_less_and_broadcastable_shapes(self):
    small = constant_op.constant([1], name="small")
    big = constant_op.constant([3, 2], name="big")
    with tensorflow_op_timer():
      test = [check_ops.assert_less(small, big)]
    with ops.control_dependencies([check_ops.assert_less(small, big)]):
      out = array_ops.identity(small)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_raises_when_less_but_non_broadcastable_shapes(self):
    small = constant_op.constant([1, 1, 1], name="small")
    big = constant_op.constant([3, 2], name="big")
    # The exception in eager and non-eager mode is different because
    # eager mode relies on shape check done as part of the C++ op, while
    # graph mode does shape checks when creating the `Operation` instance.
    with self.assertRaisesIncompatibleShapesError(
        (ValueError, errors.InvalidArgumentError)):
      with ops.control_dependencies([check_ops.assert_less(small, big)]):
        out = array_ops.identity(small)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_both_empty(self):
    larry = constant_op.constant([])
    curly = constant_op.constant([])
    with tensorflow_op_timer():
      test = [check_ops.assert_less(larry, curly)]
    with ops.control_dependencies([check_ops.assert_less(larry, curly)]):
      out = array_ops.identity(larry)
    self.evaluate(out)

  def test_returns_none_with_eager(self):
    with context.eager_mode():
      t1 = constant_op.constant([1, 2])
      t2 = constant_op.constant([3, 4])
      with tensorflow_op_timer():
        x = check_ops.assert_less(t1, t2)
      assert x is None

  def test_static_check_in_graph_mode(self):
    with ops.Graph().as_default():
      with self.assertRaisesRegex(  # pylint:disable=g-error-prone-assert-raises
          errors.InvalidArgumentError, "Custom error message"):
        check_ops.assert_less(1, 1, message="Custom error message")


class AssertLessEqualTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_equal(self):
    small = constant_op.constant([1, 2], name="small")
    with tensorflow_op_timer():
      test = [check_ops.assert_less_equal(small, small)]
    with ops.control_dependencies(
        [check_ops.assert_less_equal(small, small)]):
      out = array_ops.identity(small)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_deprecated_v1
  def test_raises_when_greater(self):
    small = constant_op.constant([1, 2], name="small")
    big = constant_op.constant([3, 4], name="big")
    with self.assertRaisesOpError(  # pylint:disable=g-error-prone-assert-raises
        "fail"):
      with ops.control_dependencies(
          [check_ops.assert_less_equal(
              big, small, message="fail")]):
        out = array_ops.identity(small)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_less_equal(self):
    small = constant_op.constant([1, 2], name="small")
    big = constant_op.constant([3, 2], name="big")
    with tensorflow_op_timer():
      test = [check_ops.assert_less_equal(small, big)]
    with ops.control_dependencies([check_ops.assert_less_equal(small, big)]):
      out = array_ops.identity(small)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_less_equal_and_broadcastable_shapes(self):
    small = constant_op.constant([1], name="small")
    big = constant_op.constant([3, 1], name="big")
    with tensorflow_op_timer():
      test = [check_ops.assert_less_equal(small, big)]
    with ops.control_dependencies([check_ops.assert_less_equal(small, big)]):
      out = array_ops.identity(small)
    self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_raises_when_less_equal_but_non_broadcastable_shapes(self):
    small = constant_op.constant([3, 1], name="small")
    big = constant_op.constant([1, 1, 1], name="big")
    # The exception in eager and non-eager mode is different because
    # eager mode relies on shape check done as part of the C++ op, while
    # graph mode does shape checks when creating the `Operation` instance.
    with self.assertRaisesRegex(  # pylint:disable=g-error-prone-assert-raises
        (errors.InvalidArgumentError, ValueError),
        (r"Incompatible shapes: \[2\] vs. \[3\]|"
         r"Dimensions must be equal, but are 2 and 3")):
      with ops.control_dependencies(
          [check_ops.assert_less_equal(small, big)]):
        out = array_ops.identity(small)
      self.evaluate(out)

  @test_util.run_in_graph_and_eager_modes
  def test_doesnt_raise_when_both_empty(self):
    larry = constant_op.constant([])
    curly = constant_op.constant([])
    with tensorflow_op_timer():
      test = [check_ops.assert_less_equal(larry, curly)]
    with ops.control_dependencies(
        [check_ops.assert_less_equal(larry, curly)]):
      out = array_ops.identity(larry)
    self.evaluate(out)

  def test_static_check_in_graph_mode(self):
    with ops.Graph().as_default():
      with self.assertRaisesRegex(  # pylint:disable=g-error-prone-assert-raises
          errors.InvalidArgumentError, "Custom error message"):
        check_ops.assert_less_equal(1, 0, message="Custom error message")

if __name__ == "__main__":
      test.main()
