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
"""Tests for vectorization of math kernels."""

from absl.testing import parameterized

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops as framework_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops.parallel_for import control_flow_ops as pfor_control_flow_ops
from tensorflow.python.ops.parallel_for.test_util import PForTestCase
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class CumProdTest(PForTestCase, parameterized.TestCase):

    def test_cum_prod(self):
        x = random_ops.random_uniform([2, 3, 4, 5])
        for axis in (1, -2, constant_op.constant(1, dtypes.int64)):
            for exclusive in (True, False):
                for reverse in (True, False):

                    # pylint: disable=cell-var-from-loop
                    def loop_fn(i):
                        a = array_ops.gather(x, i)
                        return math_ops.cumprod(
                            a, axis=axis, exclusive=exclusive, reverse=reverse)

                    # pylint: enable=cell-var-from-loop

                    self._test_loop_fn(loop_fn, 2)


if __name__ == "__main__":
    test.main()
