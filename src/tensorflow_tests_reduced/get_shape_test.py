# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for third_party.tensorflow.python.ops.ragged_tensor."""

import functools
from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import full_type_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework.type_utils import fulltypes_for_flat_tensors
from tensorflow.python.ops import array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_ragged_conversion_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_grad  # pylint: disable=unused-import
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops.ragged import ragged_concat_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_gather_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensorSpec
from tensorflow.python.ops.ragged.row_partition import RowPartition

from tensorflow.python.platform import googletest
from tensorflow.python.util import nest


def int32array(values):
  return np.array(values, dtype=np.int32)


@test_util.run_all_in_graph_and_eager_modes
class RaggedTensorTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  longMessage = True  # Property in unittest.Testcase. pylint: disable=invalid-name

  #=============================================================================
  # RaggedTensor class docstring examples
  #=============================================================================

  def testClassDocStringExamples(self):
    # From section: "Component Tensors"
    rt = RaggedTensor.from_row_splits(
        values=[3, 1, 4, 1, 5, 9, 2, 6], row_splits=[0, 4, 4, 7, 8, 8])
    self.assertAllEqual(rt, [[3, 1, 4, 1], [], [5, 9, 2], [6], []])
    del rt

    # From section: "Alternative Row-Partitioning Schemes"
    values = [3, 1, 4, 1, 5, 9, 2, 6]
    rt1 = RaggedTensor.from_row_splits(values, row_splits=[0, 4, 4, 7, 8, 8])
    rt2 = RaggedTensor.from_row_lengths(values, row_lengths=[4, 0, 3, 1, 0])
    rt3 = RaggedTensor.from_value_rowids(
        values, value_rowids=[0, 0, 0, 0, 2, 2, 2, 3], nrows=5)
    rt4 = RaggedTensor.from_row_starts(values, row_starts=[0, 4, 4, 7, 8])
    rt5 = RaggedTensor.from_row_limits(values, row_limits=[4, 4, 7, 8, 8])
    for rt in (rt1, rt2, rt3, rt4, rt5):
      self.assertAllEqual(rt, [[3, 1, 4, 1], [], [5, 9, 2], [6], []])
    del rt1, rt2, rt3, rt4, rt5

    # From section: "Multiple Ragged Dimensions"
    inner_rt = RaggedTensor.from_row_splits(
        values=[3, 1, 4, 1, 5, 9, 2, 6], row_splits=[0, 4, 4, 7, 8, 8])
    outer_rt = RaggedTensor.from_row_splits(
        values=inner_rt, row_splits=[0, 3, 3, 5])
    self.assertEqual(outer_rt.ragged_rank, 2)
    self.assertAllEqual(outer_rt,
                        [[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]])
    del inner_rt, outer_rt

    # From section: "Multiple Ragged Dimensions"
    rt = RaggedTensor.from_nested_row_splits(
        flat_values=[3, 1, 4, 1, 5, 9, 2, 6],
        nested_row_splits=([0, 3, 3, 5], [0, 4, 4, 7, 8, 8]))
    self.assertAllEqual(rt, [[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]])
    del rt

    # From section: "Uniform Inner Dimensions"
    rt = RaggedTensor.from_row_splits(
        values=array_ops.ones([5, 3]), row_splits=[0, 2, 5])
    self.assertAllEqual(
        rt, [[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]])
    self.assertEqual(rt.shape.as_list(), [2, None, 3])
    del rt

  #=============================================================================
  # RaggedTensorValue Constructor
  #=============================================================================

  def testRaggedTensorValueConstruction(self):
    values = np.array(b'a b c d e f g'.split())
    splits = np.array([0, 2, 5, 6, 6, 7], dtype=np.int64)
    splits2 = np.array([0, 3, 5], dtype=np.int64)

    # Test construction of a RaggedTensorValue with ragged_rank=1.
    rt_value = ragged_tensor_value.RaggedTensorValue(values, splits)
    self.assertEqual(rt_value.row_splits.dtype, np.int64)
    self.assertEqual(rt_value.shape, (5, None))
    self.assertLen(rt_value.nested_row_splits, 1)
    self.assertAllEqual(splits, rt_value.row_splits)
    self.assertAllEqual(values, rt_value.values)
    self.assertAllEqual(splits, rt_value.nested_row_splits[0])
    self.assertAllEqual(values, rt_value.flat_values)

    # Test construction of a RaggedTensorValue with ragged_rank=2.
    rt_value = ragged_tensor_value.RaggedTensorValue(
        values=ragged_tensor_value.RaggedTensorValue(values, splits),
        row_splits=splits2)
    self.assertEqual(rt_value.row_splits.dtype, np.int64)
    self.assertEqual(rt_value.shape, (2, None, None))
    self.assertLen(rt_value.nested_row_splits, 2)
    self.assertAllEqual(splits2, rt_value.row_splits)
    self.assertAllEqual(splits, rt_value.values.row_splits)
    self.assertAllEqual(splits2, rt_value.nested_row_splits[0])
    self.assertAllEqual(splits, rt_value.nested_row_splits[1])
    self.assertAllEqual(values, rt_value.values.values)
    self.assertAllEqual(values, rt_value.flat_values)

  #=============================================================================
  # RaggedTensor Constructor (private)
  #=============================================================================

  def testRaggedTensorConstruction(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    row_splits = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)
    rp = RowPartition.from_row_splits(row_splits)
    rt = RaggedTensor(values=values, row_partition=rp, internal=True)

    self.assertAllEqual(rt,
                        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

  def testRaggedTensorConstructionErrors(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    row_splits = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)
    rp = RowPartition.from_row_splits(row_splits)

    with self.assertRaisesRegex(ValueError,
                                'RaggedTensor constructor is private'):
      RaggedTensor(values=values, row_partition=rp)

    with self.assertRaisesRegex(
        TypeError, r'type\(values\) must be one of: Tensor, RaggedTensor'):
      RaggedTensor(values=range(7), row_partition=rp, internal=True)

    with self.assertRaisesRegex(
        TypeError, 'Argument `row_partition` must be a RowPartition'):
      RaggedTensor(
          values=values, row_partition=[0, 2, 2, 5, 6, 7], internal=True)

  #=============================================================================
  # RaggedTensor Factory Ops
  #=============================================================================

  def testFromValueRowIdsWithDerivedNRows(self):
    # nrows is known at graph creation time.
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)

    rt = RaggedTensor.from_value_rowids(values, value_rowids, validate=False)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [5, None])
    self.assertEqual(rt.ragged_rank, 1)

    rt_values = rt.values
    rt_value_rowids = rt.value_rowids()
    rt_nrows = rt.nrows()

    self.assertIs(rt_values, values)
    self.assertIs(rt_value_rowids, value_rowids)  # cached_value_rowids
    self.assertAllEqual(rt_value_rowids, value_rowids)
    self.assertAllEqual(rt_nrows, 5)
    self.assertAllEqual(rt,
                        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

  def testFromValueRowIdsWithDerivedNRowsDynamic(self):
    # nrows is not known at graph creation time.
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    value_rowids = array_ops.placeholder_with_default(value_rowids, shape=None)

    rt = RaggedTensor.from_value_rowids(values, value_rowids, validate=False)
    self.assertEqual(rt.dtype, dtypes.string)
    if context.executing_eagerly():
      self.assertEqual(rt.shape.as_list(), [5, None])
    else:
      self.assertEqual(rt.shape.as_list(), [None, None])
    self.assertEqual(rt.ragged_rank, 1)

    rt_values = rt.values
    rt_value_rowids = rt.value_rowids()
    rt_nrows = rt.nrows()

    self.assertIs(rt_values, values)
    self.assertIs(rt_value_rowids, value_rowids)  # cached_value_rowids
    self.assertAllEqual(rt_value_rowids, value_rowids)
    self.assertAllEqual(rt_nrows, 5)
    self.assertAllEqual(rt,
                        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

  def testFromValueRowIdsWithExplicitNRows(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    nrows = constant_op.constant(7, dtypes.int64)

    rt = RaggedTensor.from_value_rowids(
        values, value_rowids, nrows, validate=False)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [7, None])
    self.assertEqual(rt.ragged_rank, 1)

    rt_values = rt.values
    rt_value_rowids = rt.value_rowids()
    rt_nrows = rt.nrows()

    self.assertIs(rt_values, values)
    self.assertIs(rt_value_rowids, value_rowids)  # cached_value_rowids
    self.assertIs(rt_nrows, nrows)  # cached_nrows
    self.assertAllEqual(
        rt, [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g'], [], []])

  def testFromValueRowIdsWithExplicitNRowsEqualToDefault(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    nrows = constant_op.constant(5, dtypes.int64)

    rt = RaggedTensor.from_value_rowids(
        values, value_rowids, nrows, validate=False)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [5, None])
    self.assertEqual(rt.ragged_rank, 1)

    rt_values = rt.values
    rt_value_rowids = rt.value_rowids()
    rt_nrows = rt.nrows()

    self.assertIs(rt_values, values)
    self.assertIs(rt_value_rowids, value_rowids)  # cached_value_rowids
    self.assertIs(rt_nrows, nrows)  # cached_nrows
    self.assertAllEqual(rt_value_rowids, value_rowids)
    self.assertAllEqual(rt_nrows, nrows)
    self.assertAllEqual(rt,
                        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

  def testFromValueRowIdsWithEmptyValues(self):
    rt = RaggedTensor.from_value_rowids([], [])
    rt_nrows = rt.nrows()
    self.assertEqual(rt.dtype, dtypes.float32)
    self.assertEqual(rt.shape.as_list(), [0, None])
    self.assertEqual(rt.ragged_rank, 1)
    self.assertEqual(rt.values.shape.as_list(), [0])
    self.assertEqual(rt.value_rowids().shape.as_list(), [0])
    self.assertAllEqual(rt_nrows, 0)
    self.assertAllEqual(rt, [])

  def testFromRowSplits(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    row_splits = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)

    rt = RaggedTensor.from_row_splits(values, row_splits, validate=False)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [5, None])
    self.assertEqual(rt.ragged_rank, 1)

    rt_values = rt.values
    rt_row_splits = rt.row_splits
    rt_nrows = rt.nrows()

    self.assertIs(rt_values, values)
    self.assertIs(rt_row_splits, row_splits)
    self.assertAllEqual(rt_nrows, 5)
    self.assertAllEqual(rt,
                        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

  def testFromRowSplitsWithDifferentSplitTypes(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    splits1 = [0, 2, 2, 5, 6, 7]
    splits2 = np.array([0, 2, 2, 5, 6, 7], np.int64)
    splits3 = np.array([0, 2, 2, 5, 6, 7], np.int32)
    splits4 = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)
    splits5 = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int32)
    rt1 = RaggedTensor.from_row_splits(values, splits1)
    rt2 = RaggedTensor.from_row_splits(values, splits2)
    rt3 = RaggedTensor.from_row_splits(values, splits3)
    rt4 = RaggedTensor.from_row_splits(values, splits4)
    rt5 = RaggedTensor.from_row_splits(values, splits5)
    self.assertEqual(rt1.row_splits.dtype, dtypes.int64)
    self.assertEqual(rt2.row_splits.dtype, dtypes.int64)
    self.assertEqual(rt3.row_splits.dtype, dtypes.int32)
    self.assertEqual(rt4.row_splits.dtype, dtypes.int64)
    self.assertEqual(rt5.row_splits.dtype, dtypes.int32)

  def testFromRowSplitsWithEmptySplits(self):
    err_msg = 'row_splits tensor may not be empty'
    with self.assertRaisesRegex(ValueError, err_msg):
      RaggedTensor.from_row_splits([], [])

  def testFromRowStarts(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    row_starts = constant_op.constant([0, 2, 2, 5, 6], dtypes.int64)

    rt = RaggedTensor.from_row_starts(values, row_starts, validate=False)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [5, None])
    self.assertEqual(rt.ragged_rank, 1)

    rt_values = rt.values
    rt_row_starts = rt.row_starts()
    rt_nrows = rt.nrows()

    self.assertIs(rt_values, values)
    self.assertAllEqual(rt_nrows, 5)
    self.assertAllEqual(rt_row_starts, row_starts)
    self.assertAllEqual(rt,
                        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

  def testFromRowLimits(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    row_limits = constant_op.constant([2, 2, 5, 6, 7], dtypes.int64)

    rt = RaggedTensor.from_row_limits(values, row_limits, validate=False)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [5, None])
    self.assertEqual(rt.ragged_rank, 1)

    rt_values = rt.values
    rt_row_limits = rt.row_limits()
    rt_nrows = rt.nrows()

    self.assertIs(rt_values, values)
    self.assertAllEqual(rt_nrows, 5)
    self.assertAllEqual(rt_row_limits, row_limits)
    self.assertAllEqual(rt,
                        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

  def testFromRowLengths(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    row_lengths = constant_op.constant([2, 0, 3, 1, 1], dtypes.int64)

    rt = RaggedTensor.from_row_lengths(values, row_lengths, validate=False)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [5, None])
    self.assertEqual(rt.ragged_rank, 1)

    rt_values = rt.values
    rt_row_lengths = rt.row_lengths()
    rt_nrows = rt.nrows()

    self.assertIs(rt_values, values)
    self.assertIs(rt_row_lengths, row_lengths)  # cached_nrows
    self.assertAllEqual(rt_nrows, 5)
    self.assertAllEqual(rt_row_lengths, row_lengths)
    self.assertAllEqual(rt,
                        [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])

  def testFromRowLengthsInt32(self):
    rt = RaggedTensor.from_row_lengths([1, 2, 3, 4],
                                       constant_op.constant([1, 0, 3],
                                                            dtype=dtypes.int32))
    rt2 = RaggedTensor.from_row_lengths(rt, [2, 1, 0])
    self.assertAllEqual([2, 1, 0], rt2.row_lengths())

  def testFromUniformRowLength(self):
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    a1 = RaggedTensor.from_uniform_row_length(values, 2)
    a2 = RaggedTensor.from_uniform_row_length(values, 2, 8)
    self.assertAllEqual(
        a1,
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]])
    self.assertAllEqual(a1, a2)
    self.assertEqual(a1.shape.as_list(), [8, 2])
    self.assertEqual(a2.shape.as_list(), [8, 2])

    b1 = RaggedTensor.from_uniform_row_length(a1, 2)
    b2 = RaggedTensor.from_uniform_row_length(a1, 2, 4)
    self.assertAllEqual(b1, [[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                             [[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    self.assertAllEqual(b1, b2)
    self.assertEqual(b1.shape.as_list(), [4, 2, 2])
    self.assertEqual(b2.shape.as_list(), [4, 2, 2])

    c1 = RaggedTensor.from_uniform_row_length(b1, 2)
    c2 = RaggedTensor.from_uniform_row_length(b1, 2, 2)
    self.assertAllEqual(c1, [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                             [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])
    self.assertAllEqual(c1, c2)
    self.assertEqual(c1.shape.as_list(), [2, 2, 2, 2])
    self.assertEqual(c2.shape.as_list(), [2, 2, 2, 2])

  def testFromUniformRowLengthWithEmptyValues(self):
    empty_values = []
    a = RaggedTensor.from_uniform_row_length(empty_values, 0, nrows=10)
    self.assertEqual(a.shape.as_list(), [10, 0])

    b = RaggedTensor.from_uniform_row_length(a, 2)
    self.assertEqual(b.shape.as_list(), [5, 2, 0])

    # Make sure we avoid divide-by-zero when finding nrows for nvals=rowlen=0.
    c = RaggedTensor.from_uniform_row_length(empty_values, 0)
    self.assertEqual(c.shape.as_list(), [0, 0])
    d = RaggedTensor.from_uniform_row_length(empty_values, 0, nrows=0)
    self.assertEqual(d.shape.as_list(), [0, 0])

  def testFromUniformRowLengthWithPlaceholders(self):
    ph_values = array_ops.placeholder_with_default([1, 2, 3, 4, 5, 6], [None])
    ph_rowlen = array_ops.placeholder_with_default(3, None)
    rt1 = RaggedTensor.from_uniform_row_length(ph_values, 3)
    rt2 = RaggedTensor.from_uniform_row_length(ph_values, ph_rowlen)
    rt3 = RaggedTensor.from_uniform_row_length([1, 2, 3, 4, 5, 6], ph_rowlen)
    self.assertAllEqual(rt1, [[1, 2, 3], [4, 5, 6]])
    self.assertAllEqual(rt2, [[1, 2, 3], [4, 5, 6]])
    self.assertAllEqual(rt3, [[1, 2, 3], [4, 5, 6]])
    if context.executing_eagerly():
      self.assertEqual(rt1.shape.as_list(), [2, 3])
      self.assertEqual(rt2.shape.as_list(), [2, 3])
      self.assertEqual(rt3.shape.as_list(), [2, 3])
    else:
      self.assertEqual(rt1.shape.as_list(), [None, 3])
      self.assertEqual(rt2.shape.as_list(), [None, None])
      self.assertEqual(rt3.shape.as_list(), [None, None])

    b = RaggedTensor.from_uniform_row_length(rt1, 2)
    self.assertAllEqual(b, [[[1, 2, 3], [4, 5, 6]]])

    # Make sure we avoid divide-by-zero when finding nrows for nvals=rowlen=0.
    ph_empty_values = array_ops.placeholder_with_default(
        array_ops.zeros([0], dtypes.int64), [None])
    ph_zero = array_ops.placeholder_with_default(0, [])
    c = RaggedTensor.from_uniform_row_length(ph_empty_values, ph_zero)
    if context.executing_eagerly():
      self.assertEqual(c.shape.as_list(), [0, 0])
    else:
      self.assertEqual(c.shape.as_list(), [None, None])

  def testFromNestedValueRowIdsWithDerivedNRows(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    nested_value_rowids = [
        constant_op.constant([0, 0, 1, 3, 3], dtypes.int64),
        constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    ]

    rt = RaggedTensor.from_nested_value_rowids(values, nested_value_rowids)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [4, None, None])
    self.assertEqual(rt.ragged_rank, 2)

    rt_values = rt.values
    rt_value_rowids = rt.value_rowids()
    rt_values_values = rt_values.values
    rt_values_value_rowids = rt_values.value_rowids()

    self.assertIs(rt_values_values, values)
    self.assertAllEqual(rt_value_rowids, nested_value_rowids[0])
    self.assertAllEqual(rt_values_value_rowids, nested_value_rowids[1])
    self.assertAllEqual(
        rt, [[[b'a', b'b'], []], [[b'c', b'd', b'e']], [], [[b'f'], [b'g']]])

  def testFromNestedRowPartitions(self):
    flat_values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    nested_row_splits = [[0, 2, 3, 3, 5], [0, 2, 2, 5, 6, 7]]
    nested_row_partition = [
        RowPartition.from_row_splits(constant_op.constant(x, dtypes.int64))
        for x in nested_row_splits
    ]

    rt = RaggedTensor._from_nested_row_partitions(
        flat_values, nested_row_partition, validate=False)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [4, None, None])
    self.assertEqual(rt.ragged_rank, 2)
    self.assertAllEqual(
        rt, [[[b'a', b'b'], []], [[b'c', b'd', b'e']], [], [[b'f'], [b'g']]])

  def testFromNestedValueRowIdsWithExplicitNRows(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    nested_value_rowids = [
        constant_op.constant([0, 0, 1, 3, 3, 3], dtypes.int64),
        constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    ]
    nrows = [
        constant_op.constant(6, dtypes.int64),
        constant_op.constant(6, dtypes.int64)
    ]

    rt = RaggedTensor.from_nested_value_rowids(values, nested_value_rowids,
                                               nrows)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [6, None, None])
    self.assertEqual(rt.ragged_rank, 2)

    rt_values = rt.values
    rt_value_rowids = rt.value_rowids()
    rt_nrows = rt.nrows()
    rt_values_values = rt_values.values
    rt_values_value_rowids = rt_values.value_rowids()
    rt_values_nrows = rt_values.nrows()

    self.assertIs(rt_values_values, values)
    self.assertAllEqual(rt_value_rowids, nested_value_rowids[0])
    self.assertAllEqual(rt_values_value_rowids, nested_value_rowids[1])
    self.assertAllEqual(rt_nrows, nrows[0])
    self.assertAllEqual(rt_values_nrows, nrows[1])
    self.assertAllEqual(rt, [[[b'a', b'b'], []], [[b'c', b'd', b'e']], [],
                             [[b'f'], [b'g'], []], [], []])

  def testFromNestedValueRowIdsWithExplicitNRowsMismatch(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    nested_value_rowids = [
        constant_op.constant([0, 0, 1, 3, 3, 3], dtypes.int64),
        constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    ]
    nrows = [constant_op.constant(6, dtypes.int64)]
    with self.assertRaisesRegex(
        ValueError, 'Argument `nested_nrows` must have the same length as '
        'argument `nested_value_rowids`'):
      RaggedTensor.from_nested_value_rowids(values, nested_value_rowids, nrows)

  def testFromNestedValueRowIdsWithNonListInput(self):
    with self.assertRaisesRegex(
        TypeError, 'Argument `nested_value_rowids` must be a list of Tensors'):
      RaggedTensor.from_nested_value_rowids(
          [1, 2, 3], constant_op.constant([[0, 1, 2], [0, 1, 2]], dtypes.int64))
    with self.assertRaisesRegex(
        TypeError, 'Argument `nested_nrows` must be a list of Tensors'):
      RaggedTensor.from_nested_value_rowids([1, 2, 3], [[0, 1, 2], [0, 1, 2]],
                                            constant_op.constant([3, 3]))

  def testFromNestedRowSplits(self):
    flat_values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    nested_row_splits = [
        constant_op.constant([0, 2, 3, 3, 5], dtypes.int64),
        constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)
    ]

    rt = RaggedTensor.from_nested_row_splits(
        flat_values, nested_row_splits, validate=False)
    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [4, None, None])
    self.assertEqual(rt.ragged_rank, 2)

    rt_values = rt.values
    rt_row_splits = rt.row_splits
    rt_values_values = rt_values.values
    rt_values_row_splits = rt_values.row_splits

    self.assertIs(rt_values_values, flat_values)
    self.assertIs(rt_row_splits, nested_row_splits[0])
    self.assertIs(rt_values_row_splits, nested_row_splits[1])
    self.assertAllEqual(
        rt, [[[b'a', b'b'], []], [[b'c', b'd', b'e']], [], [[b'f'], [b'g']]])

  def testWithRowSplits(self):
    flat_values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    nested_row_splits = [
        constant_op.constant([0, 2, 3, 3, 5], dtypes.int64),
        constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)
    ]

    rt = RaggedTensor.from_nested_row_splits(
        flat_values, nested_row_splits, validate=False)

    rt = rt.with_row_splits_dtype(dtypes.int32)

    self.assertEqual(rt.dtype, dtypes.string)
    self.assertEqual(rt.shape.as_list(), [4, None, None])
    self.assertEqual(rt.ragged_rank, 2)

    rt_values = rt.values
    rt_row_splits = rt.row_splits
    rt_values_values = rt_values.values
    rt_values_row_splits = rt_values.row_splits

    self.assertAllEqual(rt_values_values, flat_values)
    self.assertAllEqual(rt_row_splits, nested_row_splits[0])
    self.assertAllEqual(rt_values_row_splits, nested_row_splits[1])
    self.assertAllEqual(
        rt, [[[b'a', b'b'], []], [[b'c', b'd', b'e']], [], [[b'f'], [b'g']]])

  def testFromNestedRowSplitsWithNonListInput(self):
    with self.assertRaisesRegex(
        TypeError, '`nested_row_splits` must be a list of Tensors'):
      RaggedTensor.from_nested_row_splits(
          [1, 2], constant_op.constant([[0, 1, 2], [0, 1, 2]], dtypes.int64))

  def testFromValueRowIdsWithBadNRows(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    nrows = constant_op.constant(5, dtypes.int64)

    with self.assertRaisesRegex(ValueError, r'Expected nrows >= 0; got -2'):
      RaggedTensor.from_value_rowids(
          values=values,
          value_rowids=array_ops.placeholder_with_default(value_rowids, None),
          nrows=-2)

    with self.assertRaisesRegex(
        ValueError, r'Expected nrows >= value_rowids\[-1\] \+ 1; got nrows=2, '
        r'value_rowids\[-1\]=4'):
      RaggedTensor.from_value_rowids(
          values=values, value_rowids=value_rowids, nrows=2)

    with self.assertRaisesRegex(
        ValueError, r'Expected nrows >= value_rowids\[-1\] \+ 1; got nrows=4, '
        r'value_rowids\[-1\]=4'):
      RaggedTensor.from_value_rowids(
          values=values, value_rowids=value_rowids, nrows=4)

    with self.assertRaisesRegex(ValueError, r'Shape \(7, 1\) must have rank 1'):
      RaggedTensor.from_value_rowids(
          values=values,
          value_rowids=array_ops.expand_dims(value_rowids, 1),
          nrows=nrows)

    with self.assertRaisesRegex(ValueError, r'Shape \(1,\) must have rank 0'):
      RaggedTensor.from_value_rowids(
          values=values,
          value_rowids=value_rowids,
          nrows=array_ops.expand_dims(nrows, 0))

  def testCondWithTensorsFromValueIds(self):
    # b/141166460
    rt = RaggedTensor.from_value_rowids([1, 2, 3], [0, 0, 2])
    c = array_ops.placeholder_with_default(True, None)
    result = control_flow_ops.cond(c, lambda: rt, lambda: rt)
    self.assertAllEqual(rt, result)

  def testGraphMismatch(self):
    if not context.executing_eagerly():
      with ops.Graph().as_default():
        values = constant_op.constant([1, 2, 3], dtypes.int64)
      with ops.Graph().as_default():
        splits = constant_op.constant([0, 2, 3], dtypes.int64)
      with self.assertRaisesRegex(ValueError,
                                  '.* must be from the same graph as .*'):
        RaggedTensor.from_row_splits(values, splits)

  @parameterized.named_parameters([
      dict(
          testcase_name='Rank0',
          tensor='a'),
      dict(
          testcase_name='Rank1',
          tensor=['a', 'b']),
  ])
  def testFromTensorRankError(self, tensor):
    with self.assertRaisesRegex(ValueError, 'must be greater than 1'):
      RaggedTensor.from_tensor(tensor)

  #=============================================================================
  # Ragged Value & Row-Partitioning Tensor Accessors
  #=============================================================================

  def testRaggedTensorAccessors_2d(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    row_splits = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    rt1 = RaggedTensor.from_row_splits(values, row_splits)
    rt2 = RaggedTensor.from_value_rowids(values, value_rowids)

    for rt in [rt1, rt2]:
      self.assertAllEqual(
          rt, [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])
      self.assertAllEqual(rt.values, [b'a', b'b', b'c', b'd', b'e', b'f', b'g'])
      self.assertEqual(rt.values.shape.dims[0].value, 7)
      self.assertAllEqual(rt.value_rowids(), [0, 0, 2, 2, 2, 3, 4])
      self.assertAllEqual(rt.nrows(), 5)
      self.assertAllEqual(rt.row_splits, [0, 2, 2, 5, 6, 7])
      self.assertAllEqual(rt.row_starts(), [0, 2, 2, 5, 6])
      self.assertAllEqual(rt.row_limits(), [2, 2, 5, 6, 7])
      self.assertAllEqual(rt.row_lengths(), [2, 0, 3, 1, 1])
      self.assertAllEqual(rt.flat_values,
                          [b'a', b'b', b'c', b'd', b'e', b'f', b'g'])
      self.assertLen(rt.nested_row_splits, 1)
      self.assertAllEqual(rt.nested_row_splits[0], [0, 2, 2, 5, 6, 7])

  def testRaggedTensorAccessors_3d_with_ragged_rank_1(self):
    values = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]]
    row_splits = constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)
    value_rowids = constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    row_lengths = constant_op.constant([2, 0, 3, 1, 1])
    rt1 = RaggedTensor.from_row_splits(values, row_splits)
    rt2 = RaggedTensor.from_value_rowids(values, value_rowids)
    rt3 = RaggedTensor.from_row_lengths(values, row_lengths)

    for rt in [rt1, rt2, rt3]:
      self.assertAllEqual(rt, [[[0, 1], [2, 3]], [], [[4, 5], [6, 7], [8, 9]],
                               [[10, 11]], [[12, 13]]])
      self.assertAllEqual(
          rt.values,
          [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]])
      self.assertEqual(rt.values.shape.dims[0].value, 7)
      self.assertAllEqual(rt.value_rowids(), [0, 0, 2, 2, 2, 3, 4])
      self.assertAllEqual(rt.nrows(), 5)
      self.assertAllEqual(rt.row_splits, [0, 2, 2, 5, 6, 7])
      self.assertAllEqual(rt.row_starts(), [0, 2, 2, 5, 6])
      self.assertAllEqual(rt.row_limits(), [2, 2, 5, 6, 7])
      self.assertAllEqual(rt.row_lengths(), [2, 0, 3, 1, 1])
      self.assertAllEqual(
          rt.row_lengths(axis=2), [[2, 2], [], [2, 2, 2], [2], [2]])
      self.assertAllEqual(
          rt.flat_values,
          [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]])
      self.assertLen(rt.nested_row_splits, 1)
      self.assertAllEqual(rt.nested_row_splits[0], [0, 2, 2, 5, 6, 7])
      self.assertLen(rt.nested_value_rowids(), 1)

      self.assertAllEqual(rt.nested_value_rowids()[0], [0, 0, 2, 2, 2, 3, 4])

  def testRaggedTensorAccessors_3d_with_ragged_rank_2(self):
    values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    nested_row_splits = [
        constant_op.constant([0, 2, 3, 3, 5], dtypes.int64),
        constant_op.constant([0, 2, 2, 5, 6, 7], dtypes.int64)
    ]
    nested_value_rowids = [
        constant_op.constant([0, 0, 1, 3, 3], dtypes.int64),
        constant_op.constant([0, 0, 2, 2, 2, 3, 4], dtypes.int64)
    ]
    rt1 = RaggedTensor.from_nested_row_splits(values, nested_row_splits)
    rt2 = RaggedTensor.from_nested_value_rowids(values, nested_value_rowids)

    for rt in [rt1, rt2]:
      self.assertAllEqual(
          rt, [[[b'a', b'b'], []], [[b'c', b'd', b'e']], [], [[b'f'], [b'g']]])
      self.assertAllEqual(
          rt.values, [[b'a', b'b'], [], [b'c', b'd', b'e'], [b'f'], [b'g']])
      self.assertEqual(rt.values.shape.dims[0].value, 5)
      self.assertAllEqual(rt.value_rowids(), [0, 0, 1, 3, 3])
      self.assertAllEqual(rt.nrows(), 4)
      self.assertAllEqual(rt.row_splits, [0, 2, 3, 3, 5])
      self.assertAllEqual(rt.row_starts(), [0, 2, 3, 3])
      self.assertAllEqual(rt.row_limits(), [2, 3, 3, 5])
      self.assertAllEqual(rt.row_lengths(), [2, 1, 0, 2])
      self.assertAllEqual(rt.flat_values,
                          [b'a', b'b', b'c', b'd', b'e', b'f', b'g'])
      self.assertLen(rt.nested_row_splits, 2)
      self.assertAllEqual(rt.nested_row_splits[0], [0, 2, 3, 3, 5])
      self.assertAllEqual(rt.nested_row_splits[1], [0, 2, 2, 5, 6, 7])
      self.assertLen(rt.nested_value_rowids(), 2)
      self.assertAllEqual(rt.nested_value_rowids()[0], [0, 0, 1, 3, 3])
      self.assertAllEqual(rt.nested_value_rowids()[1], [0, 0, 2, 2, 2, 3, 4])

  #=============================================================================
  # RaggedTensor.shape
  #=============================================================================

  def testShape(self):
    """Tests for RaggedTensor.shape."""
    rt1 = RaggedTensor.from_row_splits(b'a b c d e f g'.split(),
                                       [0, 2, 5, 6, 6, 7])
    self.assertEqual(rt1.shape.as_list(), [5, None])

    rt2 = RaggedTensor.from_row_splits(
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]],
        [0, 2, 5, 6, 6, 7])
    self.assertEqual(rt2.shape.as_list(), [5, None, 2])

    rt3 = RaggedTensor.from_row_splits(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], [0, 2, 2, 3])
    self.assertEqual(rt3.shape.as_list(), [3, None, 2, 2])

    rt4 = RaggedTensor.from_row_splits(rt3, [0, 1, 3, 3])
    self.assertEqual(rt4.shape.as_list(), [3, None, None, 2, 2])

    if not context.executing_eagerly():
      rt5 = RaggedTensor.from_row_splits(
          array_ops.placeholder(dtype=dtypes.string), [0, 2, 3, 5])
      self.assertIsNone(rt5.shape.ndims)

      rt6 = RaggedTensor.from_row_splits(
          [1, 2, 3], array_ops.placeholder(dtype=dtypes.int64))
      self.assertEqual(rt6.shape.as_list(), [None, None])

  def testGetShape(self):
    rt = RaggedTensor.from_row_splits(b'a b c d e f g'.split(),
                                      [0, 2, 5, 6, 6, 7])
    self.assertEqual(rt.shape.as_list(), rt.get_shape().as_list())

 
if __name__ == '__main__':
  googletest.main()