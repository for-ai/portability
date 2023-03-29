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

  
if __name__ == '__main__':
  googletest.main()
