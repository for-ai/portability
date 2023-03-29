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
"""Tests for vectorization of array kernels."""

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops.parallel_for import control_flow_ops as pfor_control_flow_ops
from tensorflow.python.ops.parallel_for.test_util import PForTestCase
from tensorflow.python.platform import test
from ..utils.timer_wrapper import tensorflow_op_timer


@test_util.with_eager_op_as_function
@test_util.run_all_in_graph_and_eager_modes
class ArrayTest(PForTestCase):

  def test_concat_v2(self):
    x = random_ops.random_uniform([3, 2, 3])
    y = random_ops.random_uniform([2, 3])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      with tensorflow_op_timer():
          test = array_ops.concat([x1, x1, y], axis=0)
      with tensorflow_op_timer():
          test = array_ops.concat([x1, x1, y], axis=-1)
      with tensorflow_op_timer():
          test = array_ops.concat([x1, x1, y],
                           axis=constant_op.constant(0, dtype=dtypes.int64))
      return [
          array_ops.concat([x1, x1, y], axis=0),
          array_ops.concat([x1, x1, y], axis=-1),
          array_ops.concat([x1, x1, y],
                           axis=constant_op.constant(0, dtype=dtypes.int64))
      ]

    self._test_loop_fn(loop_fn, 3)

  

if __name__ == "__main__":
  test.main()
