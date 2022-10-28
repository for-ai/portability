# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.python.ops.linalg_ops."""

import itertools

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg
from tensorflow.python.platform import test


class EyeTest(parameterized.TestCase, test.TestCase):
    @parameterized.parameters(
        itertools.product(
            # num_rows
            [0, 1, 2, 5],
            # num_columns
            [None, 0, 1, 2, 5],
            # batch_shape
            [None, [], [2], [2, 3]],
            # dtype
            [
                dtypes.int32,
                dtypes.int64,
                dtypes.float32,
                dtypes.float64,
                dtypes.complex64,
                dtypes.complex128
            ])
    )
    def test_eye_no_placeholder(self, num_rows, num_columns, batch_shape, dtype):
        eye_np = np.eye(num_rows, M=num_columns, dtype=dtype.as_numpy_dtype)
        if batch_shape is not None:
            eye_np = np.tile(eye_np, batch_shape + [1, 1])
        eye_tf = self.evaluate(linalg_ops.eye(
            num_rows,
            num_columns=num_columns,
            batch_shape=batch_shape,
            dtype=dtype))
        self.assertAllEqual(eye_np, eye_tf)

    @parameterized.parameters(
        itertools.product(
            # num_rows
            [0, 1, 2, 5],
            # num_columns
            [0, 1, 2, 5],
            # batch_shape
            [[], [2], [2, 3]],
            # dtype
            [
                dtypes.int32,
                dtypes.int64,
                dtypes.float32,
                dtypes.float64,
                dtypes.complex64,
                dtypes.complex128
            ])
    )
    @test_util.run_deprecated_v1
    def test_eye_with_placeholder(
            self, num_rows, num_columns, batch_shape, dtype):
        eye_np = np.eye(num_rows, M=num_columns, dtype=dtype.as_numpy_dtype)
        eye_np = np.tile(eye_np, batch_shape + [1, 1])
        num_rows_placeholder = array_ops.placeholder(
            dtypes.int32, name="num_rows")
        num_columns_placeholder = array_ops.placeholder(
            dtypes.int32, name="num_columns")
        batch_shape_placeholder = array_ops.placeholder(
            dtypes.int32, name="batch_shape")
        eye = linalg_ops.eye(
            num_rows_placeholder,
            num_columns=num_columns_placeholder,
            batch_shape=batch_shape_placeholder,
            dtype=dtype)
        with self.session() as sess:
            eye_tf = sess.run(
                eye,
                feed_dict={
                    num_rows_placeholder: num_rows,
                    num_columns_placeholder: num_columns,
                    batch_shape_placeholder: batch_shape
                })
        self.assertAllEqual(eye_np, eye_tf)


if __name__ == "__main__":
    test.main()
