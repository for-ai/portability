# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

import itertools

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
from ..utils.timer_wrapper import tensorflow_op_timer


default_v2_alignment = "LEFT_LEFT"
alignment_list = ["RIGHT_LEFT", "LEFT_RIGHT", "LEFT_LEFT", "RIGHT_RIGHT"]


def zip_to_first_list_length(a, b):
    if len(b) > len(a):
        return zip(a, b[:len(a)])
    return zip(a, b + [None] * (len(a) - len(b)))


def repack_diagonals(packed_diagonals,
                     diag_index,
                     num_rows,
                     num_cols,
                     align=None):
    # The original test cases are LEFT_LEFT aligned.
    if align == default_v2_alignment or align is None:
        return packed_diagonals

    align = align.split("_")
    d_lower, d_upper = diag_index
    batch_dims = packed_diagonals.ndim - (2 if d_lower < d_upper else 1)
    max_diag_len = packed_diagonals.shape[-1]
    index = (slice(None),) * batch_dims
    repacked_diagonals = np.zeros_like(packed_diagonals)

    # Aligns each diagonal row-by-row.
    for diag_index in range(d_lower, d_upper + 1):
        diag_len = min(num_rows + min(0, diag_index),
                       num_cols - max(0, diag_index))
        row_index = d_upper - diag_index
        padding_len = max_diag_len - diag_len
        left_align = (diag_index >= 0 and
                      align[0] == "LEFT") or (diag_index <= 0 and
                                              align[1] == "LEFT")
        # Prepares index tuples.
        extra_dim = tuple() if d_lower == d_upper else (row_index,)
        packed_last_dim = (slice(None),) if left_align else (
            slice(0, diag_len, 1),)
        repacked_last_dim = (slice(None),) if left_align else (slice(
            padding_len, max_diag_len, 1),)
        packed_index = index + extra_dim + packed_last_dim
        repacked_index = index + extra_dim + repacked_last_dim

        # Repacks the diagonal.
        repacked_diagonals[repacked_index] = packed_diagonals[packed_index]
    return repacked_diagonals


def repack_diagonals_in_tests(tests, align=None):
    # The original test cases are LEFT_LEFT aligned.
    if align == default_v2_alignment or align is None:
        return tests

    new_tests = dict()
    # Loops through each case.
    for diag_index, (packed_diagonals, padded_diagonals) in tests.items():
        num_rows, num_cols = padded_diagonals.shape[-2:]
        repacked_diagonals = repack_diagonals(
            packed_diagonals, diag_index, num_rows, num_cols, align=align)
        new_tests[diag_index] = (repacked_diagonals, padded_diagonals)

    return new_tests


# Test cases shared by MatrixDiagV2, MatrixDiagPartV2, and MatrixSetDiagV2.
def square_cases(align=None):
    # pyformat: disable
    mat = np.array([[[1, 2, 3, 4, 5],
                     [6, 7, 8, 9, 1],
                     [3, 4, 5, 6, 7],
                     [8, 9, 1, 2, 3],
                     [4, 5, 6, 7, 8]],
                    [[9, 1, 2, 3, 4],
                     [5, 6, 7, 8, 9],
                     [1, 2, 3, 4, 5],
                     [6, 7, 8, 9, 1],
                     [2, 3, 4, 5, 6]]])
    tests = dict()
    # tests[d_lower, d_upper] = (packed_diagonals, padded_diagonals)
    tests[-1, -1] = (np.array([[6, 4, 1, 7],
                               [5, 2, 8, 5]]),
                     np.array([[[0, 0, 0, 0, 0],
                                [6, 0, 0, 0, 0],
                                [0, 4, 0, 0, 0],
                                [0, 0, 1, 0, 0],
                                [0, 0, 0, 7, 0]],
                               [[0, 0, 0, 0, 0],
                                [5, 0, 0, 0, 0],
                                [0, 2, 0, 0, 0],
                                [0, 0, 8, 0, 0],
                                [0, 0, 0, 5, 0]]]))
    tests[-4, -3] = (np.array([[[8, 5],
                                [4, 0]],
                               [[6, 3],
                                [2, 0]]]),
                     np.array([[[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [8, 0, 0, 0, 0],
                                [4, 5, 0, 0, 0]],
                               [[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [6, 0, 0, 0, 0],
                                [2, 3, 0, 0, 0]]]))
    tests[-2, 1] = (np.array([[[2, 8, 6, 3, 0],
                               [1, 7, 5, 2, 8],
                               [6, 4, 1, 7, 0],
                               [3, 9, 6, 0, 0]],
                              [[1, 7, 4, 1, 0],
                               [9, 6, 3, 9, 6],
                               [5, 2, 8, 5, 0],
                               [1, 7, 4, 0, 0]]]),
                    np.array([[[1, 2, 0, 0, 0],
                               [6, 7, 8, 0, 0],
                               [3, 4, 5, 6, 0],
                               [0, 9, 1, 2, 3],
                               [0, 0, 6, 7, 8]],
                              [[9, 1, 0, 0, 0],
                               [5, 6, 7, 0, 0],
                               [1, 2, 3, 4, 0],
                               [0, 7, 8, 9, 1],
                               [0, 0, 4, 5, 6]]]))
    tests[2, 4] = (np.array([[[5, 0, 0],
                              [4, 1, 0],
                              [3, 9, 7]],
                             [[4, 0, 0],
                              [3, 9, 0],
                              [2, 8, 5]]]),
                   np.array([[[0, 0, 3, 4, 5],
                              [0, 0, 0, 9, 1],
                              [0, 0, 0, 0, 7],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0]],
                             [[0, 0, 2, 3, 4],
                              [0, 0, 0, 8, 9],
                              [0, 0, 0, 0, 5],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0]]]))
    # pyformat: enable
    return (mat, repack_diagonals_in_tests(tests, align))


def tall_cases(align=None):
    # pyformat: disable
    mat = np.array([[[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9],
                     [9, 8, 7],
                     [6, 5, 4]],
                    [[3, 2, 1],
                     [1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9],
                     [9, 8, 7]]])
    tests = dict()
    tests[0, 0] = (np.array([[1, 5, 9],
                             [3, 2, 6]]),
                   np.array([[[1, 0, 0],
                              [0, 5, 0],
                              [0, 0, 9],
                              [0, 0, 0]],
                             [[3, 0, 0],
                              [0, 2, 0],
                              [0, 0, 6],
                              [0, 0, 0]]]))
    tests[-4, -3] = (np.array([[[9, 5],
                                [6, 0]],
                               [[7, 8],
                                [9, 0]]]),
                     np.array([[[0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0],
                                [9, 0, 0],
                                [6, 5, 0]],
                               [[0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0],
                                [7, 0, 0],
                                [9, 8, 0]]]))
    tests[-2, -1] = (np.array([[[4, 8, 7],
                                [7, 8, 4]],
                               [[1, 5, 9],
                                [4, 8, 7]]]),
                     np.array([[[0, 0, 0],
                                [4, 0, 0],
                                [7, 8, 0],
                                [0, 8, 7],
                                [0, 0, 4]],
                               [[0, 0, 0],
                                [1, 0, 0],
                                [4, 5, 0],
                                [0, 8, 9],
                                [0, 0, 7]]]))
    tests[-2, 1] = (np.array([[[2, 6, 0],
                               [1, 5, 9],
                               [4, 8, 7],
                               [7, 8, 4]],
                              [[2, 3, 0],
                               [3, 2, 6],
                               [1, 5, 9],
                               [4, 8, 7]]]),
                    np.array([[[1, 2, 0],
                               [4, 5, 6],
                               [7, 8, 9],
                               [0, 8, 7],
                               [0, 0, 4]],
                              [[3, 2, 0],
                               [1, 2, 3],
                               [4, 5, 6],
                               [0, 8, 9],
                               [0, 0, 7]]]))
    tests[1, 2] = (np.array([[[3, 0],
                              [2, 6]],
                             [[1, 0],
                              [2, 3]]]),
                   np.array([[[0, 2, 3],
                              [0, 0, 6],
                              [0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]],
                             [[0, 2, 1],
                              [0, 0, 3],
                              [0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]]))
    # pyformat: enable
    return (mat, repack_diagonals_in_tests(tests, align))


def fat_cases(align=None):
    # pyformat: disable
    mat = np.array([[[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 1, 2, 3]],
                    [[4, 5, 6, 7],
                     [8, 9, 1, 2],
                     [3, 4, 5, 6]]])
    tests = dict()
    tests[2, 2] = (np.array([[3, 8],
                             [6, 2]]),
                   np.array([[[0, 0, 3, 0],
                              [0, 0, 0, 8],
                              [0, 0, 0, 0]],
                             [[0, 0, 6, 0],
                              [0, 0, 0, 2],
                              [0, 0, 0, 0]]]))
    tests[-2, 0] = (np.array([[[1, 6, 2],
                               [5, 1, 0],
                               [9, 0, 0]],
                              [[4, 9, 5],
                               [8, 4, 0],
                               [3, 0, 0]]]),
                    np.array([[[1, 0, 0, 0],
                               [5, 6, 0, 0],
                               [9, 1, 2, 0]],
                              [[4, 0, 0, 0],
                               [8, 9, 0, 0],
                               [3, 4, 5, 0]]]))
    tests[-1, 1] = (np.array([[[2, 7, 3],
                               [1, 6, 2],
                               [5, 1, 0]],
                              [[5, 1, 6],
                               [4, 9, 5],
                               [8, 4, 0]]]),
                    np.array([[[1, 2, 0, 0],
                               [5, 6, 7, 0],
                               [0, 1, 2, 3]],
                              [[4, 5, 0, 0],
                               [8, 9, 1, 0],
                               [0, 4, 5, 6]]]))
    tests[0, 3] = (np.array([[[4, 0, 0],
                              [3, 8, 0],
                              [2, 7, 3],
                              [1, 6, 2]],
                             [[7, 0, 0],
                              [6, 2, 0],
                              [5, 1, 6],
                              [4, 9, 5]]]),
                   np.array([[[1, 2, 3, 4],
                              [0, 6, 7, 8],
                              [0, 0, 2, 3]],
                             [[4, 5, 6, 7],
                              [0, 9, 1, 2],
                              [0, 0, 5, 6]]]))
    # pyformat: enable
    return (mat, repack_diagonals_in_tests(tests, align))


def all_tests(align=None):
    return [square_cases(align), tall_cases(align), fat_cases(align)]





class MatrixSetDiagTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testSquare(self):
        with self.session():
            v = np.array([1.0, 2.0, 3.0])
            mat = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
            mat_set_diag = np.array([[1.0, 1.0, 0.0], [1.0, 2.0, 1.0],
                                     [1.0, 1.0, 3.0]])
            timer = tensorflow_op_timer()
            with timer:
                output = array_ops.matrix_set_diag(mat, v)
                timer.gen.send(output)
            self.assertEqual((3, 3), output.get_shape())
            self.assertAllEqual(mat_set_diag, self.evaluate(output))

            # Diagonal bands.
            for align in alignment_list:
                _, tests = square_cases(align)
                for diags, (vecs, banded_mat) in tests.items():
                    mask = banded_mat[0] == 0
                    input_mat = np.random.randint(10, size=mask.shape)
                    solution = input_mat * mask + banded_mat[0]
                    timer = tensorflow_op_timer()
                    with timer:
                        output = array_ops.matrix_set_diag(
                        input_mat, vecs[0], k=diags, align=align)
                        timer.gen.send(output)
                    self.assertEqual(output.get_shape(), solution.shape)
                    self.assertAllEqual(output, solution)

    @test_util.run_deprecated_v1
    def testRectangular(self):
        with self.session():
            v = np.array([3.0, 4.0])
            mat = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
            expected = np.array([[3.0, 1.0, 0.0], [1.0, 4.0, 1.0]])
            timer = tensorflow_op_timer()
            with timer:
                output = array_ops.matrix_set_diag(mat, v)
                timer.gen.send(output)
            self.assertEqual((2, 3), output.get_shape())
            self.assertAllEqual(expected, self.evaluate(output))

            v = np.array([3.0, 4.0])
            mat = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
            expected = np.array([[3.0, 1.0], [1.0, 4.0], [1.0, 1.0]])
            timer = tensorflow_op_timer()
            with timer:
                output = array_ops.matrix_set_diag(mat, v)
                timer.gen.send(output)
            self.assertEqual((3, 2), output.get_shape())
            self.assertAllEqual(expected, self.evaluate(output))

            # Diagonal bands.
            for align in alignment_list:
                for _, tests in [tall_cases(align), fat_cases(align)]:
                    for diags, (vecs, banded_mat) in tests.items():
                        mask = banded_mat[0] == 0
                        input_mat = np.random.randint(10, size=mask.shape)
                        solution = input_mat * mask + banded_mat[0]
                        timer = tensorflow_op_timer()
                        with timer:
                            output = array_ops.matrix_set_diag(
                            input_mat, vecs[0], k=diags, align=align)
                            timer.gen.send(output)
                        self.assertEqual(output.get_shape(), solution.shape)
                        self.assertAllEqual(output, solution)

    def _testSquareBatch(self, dtype):
        with self.cached_session():
            v_batch = np.array(
                [[-1.0, 0.0, -3.0], [-4.0, -5.0, -6.0]]).astype(dtype)
            mat_batch = np.array([[[1.0, 0.0, 3.0], [0.0, 2.0, 0.0], [1.0, 0.0, 3.0]],
                                  [[4.0, 0.0, 4.0], [0.0, 5.0, 0.0],
                                   [2.0, 0.0, 6.0]]]).astype(dtype)

            mat_set_diag_batch = np.array([[[-1.0, 0.0, 3.0], [0.0, 0.0, 0.0],
                                            [1.0, 0.0, -3.0]],
                                           [[-4.0, 0.0, 4.0], [0.0, -5.0, 0.0],
                                            [2.0, 0.0, -6.0]]]).astype(dtype)
            timer = tensorflow_op_timer()
            with timer:
                output = array_ops.matrix_set_diag(mat_batch, v_batch)
                timer.gen.send(output)
            self.assertEqual((2, 3, 3), output.get_shape())
            self.assertAllEqual(mat_set_diag_batch, self.evaluate(output))

            # Diagonal bands.
            for align in alignment_list:
                _, tests = square_cases(align)
                for diags, (vecs, banded_mat) in tests.items():
                    mask = banded_mat == 0
                    input_mat = np.random.randint(
                        10, size=mask.shape).astype(dtype)
                    solution = (input_mat * mask + banded_mat).astype(dtype)
                    timer = tensorflow_op_timer()
                    with timer:
                        output = array_ops.matrix_set_diag(
                        input_mat, vecs.astype(dtype), k=diags, align=align)
                        timer.gen.send(output)
                    self.assertEqual(output.get_shape(), solution.shape)
                    self.assertAllEqual(output, solution)

    @test_util.run_deprecated_v1
    def testSquareBatch(self):
        self._testSquareBatch(np.float32)
        self._testSquareBatch(np.float64)
        self._testSquareBatch(np.int32)
        self._testSquareBatch(np.int64)
        self._testSquareBatch(np.bool_)

    @test_util.run_deprecated_v1
    def testRectangularBatch(self):
        with self.session():
            v_batch = np.array([[-1.0, -2.0], [-4.0, -5.0]])
            mat_batch = np.array([[[1.0, 0.0, 3.0], [0.0, 2.0, 0.0]],
                                  [[4.0, 0.0, 4.0], [0.0, 5.0, 0.0]]])

            mat_set_diag_batch = np.array([[[-1.0, 0.0, 3.0], [0.0, -2.0, 0.0]],
                                           [[-4.0, 0.0, 4.0], [0.0, -5.0, 0.0]]])
            timer = tensorflow_op_timer()
            with timer:
                output = array_ops.matrix_set_diag(mat_batch, v_batch)
                timer.gen.send(output)
            self.assertEqual((2, 2, 3), output.get_shape())
            self.assertAllEqual(mat_set_diag_batch, self.evaluate(output))

            # Diagonal bands.
            for align in alignment_list:
                for _, tests in [tall_cases(align), fat_cases(align)]:
                    for diags, pair in tests.items():
                        vecs, banded_mat = pair
                        mask = banded_mat == 0
                        input_mat = np.random.randint(10, size=mask.shape)
                        solution = input_mat * mask + banded_mat
                        timer = tensorflow_op_timer()
                        with timer:
                            output = array_ops.matrix_set_diag(
                            input_mat, vecs, k=diags, align=align)
                            timer.gen.send(output)
                        self.assertEqual(output.get_shape(), solution.shape)
                        self.assertAllEqual(output, solution)

    @test_util.run_deprecated_v1
    def testInvalidShape(self):
        with self.assertRaisesRegex(ValueError, "must be at least rank 2"):
            array_ops.matrix_set_diag(0, [0])
        with self.assertRaisesRegex(ValueError, "must be at least rank 1"):
            array_ops.matrix_set_diag([[0]], 0)

    @test_util.run_deprecated_v1
    def testInvalidShapeAtEval(self):
        with self.session():
            v = array_ops.placeholder(dtype=dtypes_lib.float32)
            with self.assertRaisesOpError("input must be at least 2-dim"):
                array_ops.matrix_set_diag(v, [v]).eval(feed_dict={v: 0.0})
            with self.assertRaisesOpError("diagonal must be at least 1-dim"):
                array_ops.matrix_set_diag([[v]], v).eval(feed_dict={v: 0.0})

            d = array_ops.placeholder(dtype=dtypes_lib.float32)
            with self.assertRaisesOpError(
                    "first dimensions of diagonal don't match"):
                array_ops.matrix_set_diag(v, d).eval(feed_dict={
                    v: np.zeros((2, 3, 3)),
                    d: np.ones((2, 4))
                })

    def _testGrad(self, input_shape, diag_shape, diags, align):
        with self.session():
            x = constant_op.constant(
                np.random.rand(*input_shape), dtype=dtypes_lib.float32)
            x_diag = constant_op.constant(
                np.random.rand(*diag_shape), dtype=dtypes_lib.float32)
            timer = tensorflow_op_timer()
            with timer:
                y = array_ops.matrix_set_diag(x, x_diag, k=diags, align=align)
                timer.gen.send(y)
            error_x = gradient_checker.compute_gradient_error(x,
                                                              x.get_shape().as_list(),
                                                              y,
                                                              y.get_shape().as_list())
            self.assertLess(error_x, 1e-4)
            error_x_diag = gradient_checker.compute_gradient_error(
                x_diag,
                x_diag.get_shape().as_list(), y,
                y.get_shape().as_list())
            self.assertLess(error_x_diag, 1e-4)

    @test_util.run_deprecated_v1
    def testGrad(self):
        input_shapes = [(3, 4, 4), (3, 3, 4), (3, 4, 3), (7, 4, 8, 8)]
        diag_bands = [(0, 0)]

        diag_bands.append((-1, 1))
        for input_shape, diags, align in itertools.product(input_shapes, diag_bands,
                                                           alignment_list):
            lower_diag_index, upper_diag_index = diags
            num_diags = upper_diag_index - lower_diag_index + 1
            num_diags_dim = () if num_diags == 1 else (num_diags,)
            diag_shape = input_shape[:-2] + \
                num_diags_dim + (min(input_shape[-2:]),)
            self._testGrad(input_shape, diag_shape, diags, align)

    @test_util.run_deprecated_v1
    def testGradWithNoShapeInformation(self):
        with self.session() as sess:
            v = array_ops.placeholder(dtype=dtypes_lib.float32)
            mat = array_ops.placeholder(dtype=dtypes_lib.float32)
            grad_input = array_ops.placeholder(dtype=dtypes_lib.float32)
            timer = tensorflow_op_timer()
            with timer:
                output = array_ops.matrix_set_diag(mat, v)
                timer.gen.send(output)
            grads = gradients_impl.gradients(
                output, [mat, v], grad_ys=grad_input)
            grad_input_val = np.random.rand(3, 3).astype(np.float32)
            grad_vals = sess.run(
                grads,
                feed_dict={
                    v: 2 * np.ones(3),
                    mat: np.ones((3, 3)),
                    grad_input: grad_input_val
                })
            self.assertAllEqual(np.diag(grad_input_val), grad_vals[1])
            self.assertAllEqual(grad_input_val - np.diag(np.diag(grad_input_val)),
                                grad_vals[0])



if __name__ == "__main__":
    test.main()
