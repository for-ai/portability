"""Tests for array_ops."""
import re
import time
import unittest

from absl.testing import parameterized
import numpy as np

from tensorflow.python.client import session
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow.python.platform import test as test_lib
from ..utils.timer_wrapper import tensorflow_op_timer

class SequenceMaskTest(test_util.TensorFlowTestCase):

  def testExceptions(self):
    with self.cached_session():
      with self.assertRaisesRegex(ValueError, "`maxlen` must be scalar"):
        array_ops.sequence_mask([10, 20], [10, 20])

  def testOneDimensionalWithMaxlen(self):
    timer = tensorflow_op_timer()
    with timer:
      res = array_ops.sequence_mask(constant_op.constant([1, 3, 2]), 5)
      timer.gen.send(res)
    self.assertAllEqual(res.get_shape(), [3, 5])
    self.assertAllEqual(
        res,
        [[True, False, False, False, False], [True, True, True, False, False],
         [True, True, False, False, False]])

  def testOneDimensionalDtypeWithoutMaxlen(self):
    # test dtype and default maxlen:
    timer = tensorflow_op_timer()
    with timer:
      res = array_ops.sequence_mask(
        constant_op.constant([0, 1, 4]), dtype=dtypes.float32)
      timer.gen.send(res)
    self.assertAllEqual(res.get_shape().as_list(), [3, 4])
    self.assertAllEqual(
        res, [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])

  def testOneDimensionalWithoutMaxlen(self):
    timer = tensorflow_op_timer()
    with timer:
      res = array_ops.sequence_mask(constant_op.constant([0, 1, 4]))
      timer.gen.send(res)
    self.assertAllEqual(res.get_shape().as_list(), [3, 4])
    self.assertAllEqual(res,
                        [[False, False, False, False],
                         [True, False, False, False], [True, True, True, True]])

  def testTwoDimensional(self):
    timer = tensorflow_op_timer()
    with timer:
      res = array_ops.sequence_mask(constant_op.constant([[1, 3, 2]]), 5)
      timer.gen.send(res)
    self.assertAllEqual(res.get_shape(), [1, 3, 5])
    self.assertAllEqual(
        res,
        [[[True, False, False, False, False], [True, True, True, False, False],
          [True, True, False, False, False]]])

    # test dtype and default maxlen:
    timer = tensorflow_op_timer()
    with timer:
      res = array_ops.sequence_mask(
        constant_op.constant([[0, 1, 4], [1, 2, 3]]), dtype=dtypes.float32)
      timer.gen.send(res)
    self.assertAllEqual(res.get_shape().as_list(), [2, 3, 4])
    self.assertAllEqual(
        res,
        [[[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
         [[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]]])

  def testDtypes(self):

    def check_dtypes(lengths_dtype, maxlen_dtype):
      timer = tensorflow_op_timer()
      with timer:
        res = array_ops.sequence_mask(
          constant_op.constant([1, 3, 2], dtype=lengths_dtype),
          constant_op.constant(5, dtype=maxlen_dtype))
        timer.gen.send(res)
      self.assertAllEqual(res.get_shape(), [3, 5])
      self.assertAllEqual(
          res,
          [[True, False, False, False, False], [True, True, True, False, False],
           [True, True, False, False, False]])

    check_dtypes(dtypes.int32, dtypes.int32)
    check_dtypes(dtypes.int32, dtypes.int64)
    check_dtypes(dtypes.int64, dtypes.int32)
    check_dtypes(dtypes.int64, dtypes.int64)

  def testOutputDtype(self):

    def check_output_dtype(output_dtype):
      timer = tensorflow_op_timer()
      with timer:
        res = self.evaluate(
          array_ops.sequence_mask(
              constant_op.constant([1, 3, 2], dtype=dtypes.int32),
              constant_op.constant(5, dtype=dtypes.int32),
              dtype=output_dtype))
        timer.gen.send(res)
      self.assertAllEqual(
          res,
          self.evaluate(
              math_ops.cast([[True, False, False, False, False],
                             [True, True, True, False, False],
                             [True, True, False, False, False]], output_dtype)))

    check_output_dtype(dtypes.bool)
    check_output_dtype("bool")
    check_output_dtype(np.bool_)
    check_output_dtype(dtypes.int32)
    check_output_dtype("int32")
    check_output_dtype(np.int32)
    check_output_dtype(dtypes.float32)
    check_output_dtype("float32")
    check_output_dtype(np.float32)
    check_output_dtype(dtypes.int64)
    check_output_dtype("float64")
    check_output_dtype(np.float64)



if __name__ == "__main__":
  test_lib.main()
