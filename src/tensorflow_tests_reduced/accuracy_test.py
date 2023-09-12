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
"""Tests for metrics."""

import functools
import math

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
import tensorflow.python.ops.data_flow_grad  # pylint: disable=unused-import
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from ..utils.timer_wrapper import tensorflow_op_timer,  tensorflow_timer
import pytest

NAN = float('nan')


def _enqueue_vector(sess, queue, values, shape=None):
  if not shape:
    shape = (1, len(values))
  dtype = queue.dtypes[0]
  sess.run(
      queue.enqueue(constant_op.constant(
          values, dtype=dtype, shape=shape)))


def _binary_2d_label_to_2d_sparse_value(labels):
  """Convert dense 2D binary indicator to sparse ID.

  Only 1 values in `labels` are included in result.

  Args:
    labels: Dense 2D binary indicator, shape [batch_size, num_classes].

  Returns:
    `SparseTensorValue` of shape [batch_size, num_classes], where num_classes
    is the number of `1` values in each row of `labels`. Values are indices
    of `1` values along the last dimension of `labels`.
  """
  indices = []
  values = []
  batch = 0
  for row in labels:
    label = 0
    xi = 0
    for x in row:
      if x == 1:
        indices.append([batch, xi])
        values.append(label)
        xi += 1
      else:
        assert x == 0
      label += 1
    batch += 1
  shape = [len(labels), len(labels[0])]
  return sparse_tensor.SparseTensorValue(
      np.array(indices, np.int64),
      np.array(values, np.int64), np.array(shape, np.int64))


def _binary_2d_label_to_1d_sparse_value(labels):
  """Convert dense 2D binary indicator to sparse ID.

  Only 1 values in `labels` are included in result.

  Args:
    labels: Dense 2D binary indicator, shape [batch_size, num_classes]. Each
    row must contain exactly 1 `1` value.

  Returns:
    `SparseTensorValue` of shape [batch_size]. Values are indices of `1` values
    along the last dimension of `labels`.

  Raises:
    ValueError: if there is not exactly 1 `1` value per row of `labels`.
  """
  indices = []
  values = []
  batch = 0
  for row in labels:
    label = 0
    xi = 0
    for x in row:
      if x == 1:
        indices.append([batch])
        values.append(label)
        xi += 1
      else:
        assert x == 0
      label += 1
    batch += 1
  if indices != [[i] for i in range(len(labels))]:
    raise ValueError('Expected 1 label/example, got %s.' % indices)
  shape = [len(labels)]
  return sparse_tensor.SparseTensorValue(
      np.array(indices, np.int64),
      np.array(values, np.int64), np.array(shape, np.int64))


def _binary_3d_label_to_sparse_value(labels):
  """Convert dense 3D binary indicator tensor to sparse tensor.

  Only 1 values in `labels` are included in result.

  Args:
    labels: Dense 2D binary indicator tensor.

  Returns:
    `SparseTensorValue` whose values are indices along the last dimension of
    `labels`.
  """
  indices = []
  values = []
  for d0, labels_d0 in enumerate(labels):
    for d1, labels_d1 in enumerate(labels_d0):
      d2 = 0
      for class_id, label in enumerate(labels_d1):
        if label == 1:
          values.append(class_id)
          indices.append([d0, d1, d2])
          d2 += 1
        else:
          assert label == 0
  shape = [len(labels), len(labels[0]), len(labels[0][0])]
  return sparse_tensor.SparseTensorValue(
      np.array(indices, np.int64),
      np.array(values, np.int64), np.array(shape, np.int64))


def _assert_nan(test_case, actual):
  test_case.assertTrue(math.isnan(actual), 'Expected NAN, got %s.' % actual)


def _assert_metric_variables(test_case, expected):
  test_case.assertEqual(
      set(expected), set(v.name for v in variables.local_variables()))
  test_case.assertEqual(
      set(expected),
      set(v.name for v in ops.get_collection(ops.GraphKeys.METRIC_VARIABLES)))


def _test_values(shape):
  return np.reshape(np.cumsum(np.ones(shape)), newshape=shape)


class AccuracyTest(test.TestCase):

  def setUp(self):
    ops.reset_default_graph()

  @test_util.run_deprecated_v1
  def testVars(self):
    timer = tensorflow_op_timer()
    with timer:
      metrics.accuracy(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        name='my_accuracy')
      timer.gen.send(None)
    _assert_metric_variables(self,
                             ('my_accuracy/count:0', 'my_accuracy/total:0'))

  @test_util.run_deprecated_v1
  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    timer = tensorflow_op_timer()
    with timer:
      mean, _ = metrics.accuracy(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        metrics_collections=[my_collection_name])
      timer.gen.send(None)
    self.assertListEqual(ops.get_collection(my_collection_name), [mean])

  @test_util.run_deprecated_v1
  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    timer = tensorflow_op_timer()
    with timer:
      _, update_op = metrics.accuracy(
        predictions=array_ops.ones((10, 1)),
        labels=array_ops.ones((10, 1)),
        updates_collections=[my_collection_name])
      timer.gen.send(None)
    self.assertListEqual(ops.get_collection(my_collection_name), [update_op])

  @test_util.run_deprecated_v1
  def testPredictionsAndLabelsOfDifferentSizeRaisesValueError(self):
    predictions = array_ops.ones((10, 3))
    labels = array_ops.ones((10, 4))
    with self.assertRaises(ValueError):
      metrics.accuracy(labels, predictions)

  @test_util.run_deprecated_v1
  def testPredictionsAndWeightsOfDifferentSizeRaisesValueError(self):
    predictions = array_ops.ones((10, 3))
    labels = array_ops.ones((10, 3))
    weights = array_ops.ones((9, 3))
    with self.assertRaises(ValueError):
      metrics.accuracy(labels, predictions, weights)

  # @test_util.run_deprecated_v1
  # def testValueTensorIsIdempotent(self):
  #   predictions = random_ops.random_uniform(
  #       (10, 3), maxval=3, dtype=dtypes_lib.int64, seed=1)
  #   labels = random_ops.random_uniform(
  #       (10, 3), maxval=3, dtype=dtypes_lib.int64, seed=1)
  #   timer = tensorflow_op_timer()
  #   with timer:
  #     accuracy, update_op = metrics.accuracy(labels, predictions)
  #     timer.gen.send(None)

  #   with self.cached_session():
  #     self.evaluate(variables.local_variables_initializer())

  #     # Run several updates.
  #     for _ in range(10):
  #       self.evaluate(update_op)

  #     # Then verify idempotency.
  #     initial_accuracy = self.evaluate(accuracy)
  #     for _ in range(10):
  #       self.assertEqual(initial_accuracy, self.evaluate(accuracy))

  @test_util.run_deprecated_v1
  def testMultipleUpdates(self):
    with self.cached_session() as sess:
      # Create the queue that populates the predictions.
      preds_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 1))
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [2])
      _enqueue_vector(sess, preds_queue, [1])
      predictions = preds_queue.dequeue()

      # Create the queue that populates the labels.
      labels_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 1))
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [2])
      labels = labels_queue.dequeue()
      timer = tensorflow_op_timer()
      with timer:
        accuracy, update_op = metrics.accuracy(labels, predictions)
        timer.gen.send(None)

      self.evaluate(variables.local_variables_initializer())
      for _ in range(3):
        self.evaluate(update_op)
      self.assertEqual(0.5, self.evaluate(update_op))
      self.assertEqual(0.5, self.evaluate(accuracy))

  @test_util.run_deprecated_v1
  def testEffectivelyEquivalentSizes(self):
    predictions = array_ops.ones((40, 1))
    labels = array_ops.ones((40,))
    with self.cached_session():
      timer = tensorflow_op_timer()
      with timer:
        accuracy, update_op = metrics.accuracy(labels, predictions)
        timer.gen.send(None)

      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(1.0, self.evaluate(update_op))
      self.assertEqual(1.0, self.evaluate(accuracy))

  @test_util.run_deprecated_v1
  def testEffectivelyEquivalentSizesWithScalarWeight(self):
    predictions = array_ops.ones((40, 1))
    labels = array_ops.ones((40,))
    with self.cached_session():
      timer = tensorflow_op_timer()
      with timer:
        accuracy, update_op = metrics.accuracy(labels, predictions, weights=2.0)
        timer.gen.send(None)

      self.evaluate(variables.local_variables_initializer())
      self.assertEqual(1.0, self.evaluate(update_op))
      self.assertEqual(1.0, self.evaluate(accuracy))

  @test_util.run_deprecated_v1
  def testEffectivelyEquivalentSizesWithStaticShapedWeight(self):
    predictions = ops.convert_to_tensor([1, 1, 1])  # shape 3,
    labels = array_ops.expand_dims(ops.convert_to_tensor([1, 0, 0]),
                                   1)  # shape 3, 1
    weights = array_ops.expand_dims(ops.convert_to_tensor([100, 1, 1]),
                                    1)  # shape 3, 1

    with self.cached_session():
      timer = tensorflow_op_timer()
      with timer:
        accuracy, update_op = metrics.accuracy(labels, predictions, weights)
        timer.gen.send(None)

      self.evaluate(variables.local_variables_initializer())
      # if streaming_accuracy does not flatten the weight, accuracy would be
      # 0.33333334 due to an intended broadcast of weight. Due to flattening,
      # it will be higher than .95
      self.assertGreater(self.evaluate(update_op), .95)
      self.assertGreater(self.evaluate(accuracy), .95)

  @test_util.run_deprecated_v1
  def testEffectivelyEquivalentSizesWithDynamicallyShapedWeight(self):
    predictions = ops.convert_to_tensor([1, 1, 1])  # shape 3,
    labels = array_ops.expand_dims(ops.convert_to_tensor([1, 0, 0]),
                                   1)  # shape 3, 1

    weights = [[100], [1], [1]]  # shape 3, 1
    weights_placeholder = array_ops.placeholder(
        dtype=dtypes_lib.int32, name='weights')
    feed_dict = {weights_placeholder: weights}

    with self.cached_session():
      timer = tensorflow_op_timer()
      with timer:
        accuracy, update_op = metrics.accuracy(labels, predictions,
                                             weights_placeholder)
        timer.gen.send(None)

      self.evaluate(variables.local_variables_initializer())
      # if streaming_accuracy does not flatten the weight, accuracy would be
      # 0.33333334 due to an intended broadcast of weight. Due to flattening,
      # it will be higher than .95
      self.assertGreater(update_op.eval(feed_dict=feed_dict), .95)
      self.assertGreater(accuracy.eval(feed_dict=feed_dict), .95)

  @test_util.run_deprecated_v1
  def testMultipleUpdatesWithWeightedValues(self):
    with self.cached_session() as sess:
      # Create the queue that populates the predictions.
      preds_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 1))
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [2])
      _enqueue_vector(sess, preds_queue, [1])
      predictions = preds_queue.dequeue()

      # Create the queue that populates the labels.
      labels_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.float32, shapes=(1, 1))
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [2])
      labels = labels_queue.dequeue()

      # Create the queue that populates the weights.
      weights_queue = data_flow_ops.FIFOQueue(
          4, dtypes=dtypes_lib.int64, shapes=(1, 1))
      _enqueue_vector(sess, weights_queue, [1])
      _enqueue_vector(sess, weights_queue, [1])
      _enqueue_vector(sess, weights_queue, [0])
      _enqueue_vector(sess, weights_queue, [0])
      weights = weights_queue.dequeue()
      timer = tensorflow_op_timer()
      with timer:
        accuracy, update_op = metrics.accuracy(labels, predictions, weights)
        timer.gen.send(None)

      self.evaluate(variables.local_variables_initializer())
      for _ in range(3):
        self.evaluate(update_op)
      self.assertEqual(1.0, self.evaluate(update_op))
      self.assertEqual(1.0, self.evaluate(accuracy))



if __name__ == '__main__':
  test.main()
