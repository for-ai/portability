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
"""Functional tests for binary coefficient-wise operations."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_grad  # pylint: disable=unused-import
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


def _ADD(x, y): return x + y
def _SUB(x, y): return x - y
def _MUL(x, y): return x * y
def _POW(x, y): return x**y
def _TRUEDIV(x, y): return x / y
def _FLOORDIV(x, y): return x // y
def _MOD(x, y): return x % y


class BinaryOpTest(test.TestCase):
    def _compareCpu(self, x, y, np_func, tf_func, also_compare_variables=False):
        np_ans = np_func(x, y)
        with test_util.force_cpu():
            inx = ops.convert_to_tensor(x)
            iny = ops.convert_to_tensor(y)
            out = tf_func(inx, iny)
            tf_cpu = self.evaluate(out)
            # Test that the op takes precedence over numpy operators.
            np_left = self.evaluate(tf_func(x, iny))
            np_right = self.evaluate(tf_func(inx, y))

            if also_compare_variables:
                var_x = variables.Variable(x)
                var_y = variables.Variable(y)
                self.evaluate(variables.global_variables_initializer())
                print(type(x), type(y), type(var_x), type(var_y))
                print(type(tf_func(x, var_y)), type(tf_func(var_x, y)))
                np_var_left = self.evaluate(tf_func(x, var_y))
                np_var_right = self.evaluate(tf_func(var_x, y))

        if np_ans.dtype != np.object_:
            self.assertAllClose(np_ans, tf_cpu)
            self.assertAllClose(np_ans, np_left)
            self.assertAllClose(np_ans, np_right)
            if also_compare_variables:
                self.assertAllClose(np_ans, np_var_left)
                self.assertAllClose(np_ans, np_var_right)
        self.assertShapeEqual(np_ans, out)

    def _compareGpu(self, x, y, np_func, tf_func):
        np_ans = np_func(x, y)
        with test_util.use_gpu():
            inx = ops.convert_to_tensor(x)
            iny = ops.convert_to_tensor(y)
            out = tf_func(inx, iny)
            tf_gpu = self.evaluate(out)
        self.assertAllClose(np_ans, tf_gpu)
        self.assertShapeEqual(np_ans, out)

    def testAtan2SpecialValues(self):
        x1l, x2l = zip((+0.0, +0.0), (+0.0, -0.0), (-0.0, +0.0), (-0.0, -0.0),
                       (1.2345, float("inf")), (1.2345, -float("inf")),
                       (-4.321, float("inf")), (-4.125, -float("inf")),
                       (float("inf"), float("inf")), (float("inf"), -float("inf")),
                       (-float("inf"), float("inf")),
                       (-float("inf"), -float("inf")))
        for dtype in np.float32, np.float64:
            x1 = np.array(x1l).astype(dtype)
            x2 = np.array(x2l).astype(dtype)
            self._compareCpu(x1, x2, np.arctan2, math_ops.atan2)
            self._compareGpu(x1, x2, np.arctan2, math_ops.atan2)


if __name__ == "__main__":
    test.main()
