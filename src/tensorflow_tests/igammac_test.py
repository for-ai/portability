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
class MathTest(PForTestCase, parameterized.TestCase):

    def test_binary_cwise_ops(self):
        # Enable tensor equality to test `equal` and `not_equal` ops below.
        default_equality = framework_ops.Tensor._USE_EQUALITY
        framework_ops.enable_tensor_equality()
        try:
            logical_ops = [
                math_ops.logical_and, math_ops.logical_or, math_ops.logical_xor
            ]

            # Wrapper functions restricting the range of inputs of zeta and polygamma.
            def safe_polygamma(x, y):
                return math_ops.polygamma(
                    math_ops.round(clip_ops.clip_by_value(y, 1, 10)), x * x + 1)

            def safe_zeta(x, y):
                return math_ops.zeta(x * x + 1, y * y)

            float_ops = [
                math_ops.igammac,
            ]
            # FloorDiv fails on XLA due floor's discontinuities exacerbating small
            # division differences.
            if not test_util.is_xla_enabled():
                float_ops += [math_ops.floor_div]
                # TODO(b/168912036): Re-enable once GPU + XLA issues for Zeta are
                # resolved.
                if not test_util.is_gpu_available():
                    float_ops += [safe_zeta]
            for op in logical_ops + float_ops:
                x = random_ops.random_uniform([7, 3, 5])
                y = random_ops.random_uniform([3, 5])
                if op in logical_ops:
                    x = x > 0
                    y = y > 0

                output_dtypes = []

                # pylint: disable=cell-var-from-loop
                def loop_fn(i):
                    x1 = array_ops.gather(x, i)
                    y1 = array_ops.gather(y, i)
                    outputs = [op(x, y), op(x1, y), op(
                        x, y1), op(x1, y1), op(x1, x1)]
                    del output_dtypes[:]
                    output_dtypes.extend(t.dtype for t in outputs)
                    return outputs

                # pylint: enable=cell-var-from-loop

                self._test_loop_fn(loop_fn, 3)
        finally:
            if not default_equality:
                framework_ops.disable_tensor_equality()


if __name__ == "__main__":
    test.main()
