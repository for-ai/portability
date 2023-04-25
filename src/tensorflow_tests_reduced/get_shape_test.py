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
from ..utils.timer_wrapper import tensorflow_op_timer


def int32array(values):
  return np.array(values, dtype=np.int32)


@test_util.run_all_in_graph_and_eager_modes
class RaggedTensorTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  longMessage = True  # Property in unittest.Testcase. pylint: disable=invalid-name

  #=============================================================================
  # RaggedTensor class docstring examples
  #=============================================================================

  def testGetShape(self):
    rt = RaggedTensor.from_row_splits(b'a b c d e f g'.split(),
                                      [0, 2, 5, 6, 6, 7])
    timer = tensorflow_op_timer()
    with timer:
      test = rt.get_shape().as_list()
      timer.gen.send(test)
    self.assertEqual(rt.shape.as_list(), rt.get_shape().as_list())

 
if __name__ == '__main__':
  googletest.main()