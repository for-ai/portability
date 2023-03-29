# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.tf.scatter_nd."""

import functools

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

GRADIENT_TESTS_DTYPES = (dtypes.float16, dtypes.float32, dtypes.float64)
from ..utils.timer_wrapper import tensorflow_op_timer





class ScatterNdTensorTest(test.TestCase):

    @test_util.run_in_graph_and_eager_modes
    def testUpdateAddSub(self):
        for dtype in (dtypes.int32, dtypes.float32):
            indices = constant_op.constant([[4], [3], [1], [7]])
            updates = constant_op.constant([9, 10, 11, 12], dtype=dtype)
            t = array_ops.ones([8], dtype=dtype)
            with tensorflow_op_timer():
                assigned = array_ops.tensor_scatter_update(t, indices, updates)
            added = array_ops.tensor_scatter_add(t, indices, updates)
            subbed = array_ops.tensor_scatter_sub(t, indices, updates)

            self.assertAllEqual(assigned,
                                constant_op.constant([1, 11, 1, 10, 9, 1, 1, 12]))
            self.assertAllEqual(added,
                                constant_op.constant([1, 12, 1, 11, 10, 1, 1, 13]))
            self.assertAllEqual(subbed,
                                constant_op.constant([1, -10, 1, -9, -8, 1, 1, -11]))

    def testUpdateAddSubGradients(self):
        with self.cached_session():
            indices = constant_op.constant([[3], [1]])
            updates = constant_op.constant([9, 10], dtype=dtypes.float32)
            x = array_ops.ones([4], dtype=dtypes.float32)

            theoretical, numerical = gradient_checker_v2.compute_gradient(
                lambda x: array_ops.tensor_scatter_update(x, indices, updates), [x])
            self.assertAllClose(theoretical, numerical, 5e-4, 5e-4)
            theoretical, numerical = gradient_checker_v2.compute_gradient(
                lambda x: array_ops.tensor_scatter_add(x, indices, updates), [x])
            self.assertAllClose(theoretical, numerical, 5e-4, 5e-4)
            theoretical, numerical = gradient_checker_v2.compute_gradient(
                lambda x: array_ops.tensor_scatter_sub(x, indices, updates), [x])
            self.assertAllClose(theoretical, numerical, 5e-4, 5e-4)

            theoretical, numerical = gradient_checker_v2.compute_gradient(
                lambda updates: array_ops.tensor_scatter_update(
                    x, indices, updates),
                [updates])
            self.assertAllClose(theoretical, numerical, 5e-4, 5e-4)
            theoretical, numerical = gradient_checker_v2.compute_gradient(
                lambda updates: array_ops.tensor_scatter_add(
                    x, indices, updates),
                [updates])
            self.assertAllClose(theoretical, numerical, 5e-4, 5e-4)
            theoretical, numerical = gradient_checker_v2.compute_gradient(
                lambda updates: array_ops.tensor_scatter_sub(
                    x, indices, updates),
                [updates])
            self.assertAllClose(theoretical, numerical, 5e-4, 5e-4)

    @test_util.run_in_graph_and_eager_modes
    def testUpdateMinMax(self):
        for dtype in (dtypes.int32, dtypes.float32):
            indices = constant_op.constant([[4], [3], [1], [7]])
            updates = constant_op.constant([0, 2, -1, 2], dtype=dtype)
            t = array_ops.ones([8], dtype=dtype)
            with tensorflow_op_timer():
                assigned = array_ops.tensor_scatter_update(t, indices, updates)
            min_result = array_ops.tensor_scatter_min(t, indices, updates)
            max_result = array_ops.tensor_scatter_max(t, indices, updates)

            self.assertAllEqual(assigned,
                                constant_op.constant([1, -1, 1, 2, 0, 1, 1, 2]))
            self.assertAllEqual(min_result,
                                constant_op.constant([1, -1, 1, 1, 0, 1, 1, 1]))
            self.assertAllEqual(max_result,
                                constant_op.constant([1, 1, 1, 2, 1, 1, 1, 2]))


    def testTensorScatterUpdateWithForwarding(self):
        for dtype in (dtypes.int32, dtypes.float32):

            @def_function.function
            def _TestFn():
                indices = constant_op.constant([[4], [3], [1], [7]])
                updates = constant_op.constant(
                    [9, 10, 11, 12], dtype=dtype)  # pylint: disable=cell-var-from-loop
                t = array_ops.ones(
                    [8], dtype=dtype)  # pylint: disable=cell-var-from-loop
                with tensorflow_op_timer():
                    test = array_ops.tensor_scatter_update(t, indices, updates)

                return array_ops.tensor_scatter_update(t, indices, updates)

            self.assertAllEqual(_TestFn(), [1, 11, 1, 10, 9, 1, 1, 12])

    @test_util.run_in_graph_and_eager_modes
    def testTensorScatterUpdateWithStrings(self):
        indices = constant_op.constant([[4], [3], [1], [7]])
        updates = constant_op.constant(["there", "there", "there", "12"],
                                       dtype=dtypes.string)
        tensor = constant_op.constant([
            "hello", "hello", "hello", "hello", "hello", "hello", "hello", "hello"
        ],
            dtype=dtypes.string)
        with tensorflow_op_timer():
            updated = array_ops.tensor_scatter_update(tensor, indices, updates)

        self.assertAllEqual(
            updated,
            constant_op.constant([
                "hello", "there", "hello", "there", "there", "hello", "hello", "12"
            ]))

    @test_util.run_in_graph_and_eager_modes
    def testUpdateRepeatedIndices1D(self):
        if test_util.is_gpu_available():
            self.skipTest(
                "Duplicate indices scatter is non-deterministic on GPU")
        a = array_ops.zeros([10, 1])
        with tensorflow_op_timer():
            b = array_ops.tensor_scatter_update(a, [[5], [5]], [[4], [8]])
        self.assertAllEqual(
            b,
            constant_op.constant([[0.], [0.], [0.], [0.], [0.], [8.], [0.], [0.],
                                  [0.], [0.]]))

    @test_util.run_in_graph_and_eager_modes
    def testUpdateRepeatedIndices2D(self):
        if test_util.is_gpu_available():
            self.skipTest(
                "Duplicate indices scatter is non-deterministic on GPU")
        a = array_ops.zeros([10, 10])
        with tensorflow_op_timer():
            b = array_ops.tensor_scatter_update(
            a, [[5], [6], [6]],
            [math_ops.range(10),
             math_ops.range(11, 21),
             math_ops.range(10, 20)])
        self.assertAllEqual(
            b[6],
            constant_op.constant([10., 11., 12., 13., 14., 15., 16., 17., 18.,
                                  19.]))


class ScatterNdTensorDeterminismTest(ScatterNdTensorTest):

    def setUp(self):
        super().setUp()
        config.enable_op_determinism()

    def tearDown(self):
        super().tearDown()
        config.disable_op_determinism()

    def testDeterminism(self):
        a = array_ops.zeros([1])
        indices = array_ops.zeros([100000, 1], dtypes.int32)
        values = np.random.randn(100000)
        with tensorflow_op_timer():
            test = array_ops.tensor_scatter_update(a, indices, values)
        val = self.evaluate(
            array_ops.tensor_scatter_update(a, indices, values))
        for _ in range(5):
            with tensorflow_op_timer():
                test = array_ops.tensor_scatter_update(a, indices, values)
            val2 = self.evaluate(
                array_ops.tensor_scatter_update(a, indices, values))
            self.assertAllEqual(val, val2)


if __name__ == "__main__":
    test.main()
