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


def _TRUEDIV(x, y): return x / y


class TrueDivTest(test.TestCase):
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

    def _compareBoth(self, x, y, np_func, tf_func, also_compare_variables=False):
        self._compareCpu(x, y, np_func, tf_func, also_compare_variables)
        if x.dtype in (np.float16, np.float32, np.float64, np.complex64,
                       np.complex128):
            if tf_func not in (_FLOORDIV, math_ops.floordiv, math_ops.zeta,
                               math_ops.polygamma):
                self._compareGradientX(x, y, np_func, tf_func)
                self._compareGradientY(x, y, np_func, tf_func)
            if tf_func in (math_ops.zeta, math_ops.polygamma):
                # These methods only support gradients in the second parameter
                self._compareGradientY(x, y, np_func, tf_func)
            self._compareGpu(x, y, np_func, tf_func)

    def testInt32Basic(self):
        x = np.arange(1, 13, 2).reshape(1, 3, 2).astype(np.int32)
        y = np.arange(1, 7, 1).reshape(1, 3, 2).astype(np.int32)
        self._compareBoth(x, y, np.true_divide, math_ops.truediv)


if __name__ == "__main__":
    test.main()
