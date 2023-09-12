# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensor_array_ops."""

import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.platform import test
from ..utils.timer_wrapper import tensorflow_op_timer


class TensorArrayOpsTest(test.TestCase):
    @test_util.run_v2_only
    def test_concat(self):
        values = tensor_array_ops.TensorArray(
            size=4, dtype=dtypes.string, element_shape=[None], infer_shape=False
        )
        a = constant_op.constant(["a", "b", "c"], dtypes.string)
        b = constant_op.constant(["c", "d", "e"], dtypes.string)
        values = (
            (values.write(0, a).write(1, constant_op.constant([], dtypes.string)))
            .write(2, b)
            .write(3, constant_op.constant([], dtypes.string))
        )
        timer = tensorflow_op_timer()
        with timer:
            result = values.concat()
            timer.gen.send(result)
        self.assertAllEqual(result, [b"a", b"b", b"c", b"c", b"d", b"e"])
