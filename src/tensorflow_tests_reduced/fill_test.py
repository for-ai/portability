# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for wrapping an eager op in a call op at runtime."""
import time

from tensorflow.python.data.experimental.ops import prefetching_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import benchmarks_test_base
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import critical_section_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_map_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import tf_inspect
from ..utils.timer_wrapper import tensorflow_op_timer


# @test_util.with_eager_op_as_function
class RunEagerOpAsFunctionTest(test.TestCase):

    def setUp(self):
        super().setUp()
        self._m_2_by_2 = random_ops.random_uniform((2, 2))

    def testArrayFill(self):
        timer = tensorflow_op_timer()
        with timer:
            test = array_ops.fill(
            constant_op.constant([2], dtype=dtypes.int64), constant_op.constant(1))
            timer.gen.send(test)


if __name__ == "__main__":
    test.main()
