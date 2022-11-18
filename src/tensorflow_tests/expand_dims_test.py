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
"""Tests for various tensorflow.ops.tf."""

from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import importer
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.platform import test


# TODO(zongheng): it'd be great to factor out this function and various random
# SparseTensor gen funcs.
def _sparsify(x, thresh=0.5, index_dtype=np.int64):
  x[x < thresh] = 0

  non_zero = np.where(x)
  x_indices = np.vstack(non_zero).astype(index_dtype).T
  x_values = x[non_zero]
  x_shape = x.shape

  return sparse_tensor.SparseTensor(
      indices=x_indices, values=x_values, dense_shape=x_shape), len(x_values)


class ShapeOpsTest(test.TestCase):

  def _compareShape(self, x, use_gpu=False):
    np_ans = np.array(np.shape(x))
    with self.cached_session(use_gpu=use_gpu):
      tf_ans = array_ops.shape(x)
      tf_ans_64 = array_ops.shape(x, out_type=dtypes.int64)
      result = self.evaluate(tf_ans)
      result_64 = self.evaluate(tf_ans_64)
    self.assertAllEqual(np_ans, result)
    self.assertAllEqual(np_ans, result_64)
    self.assertShapeEqual(np_ans, tf_ans)

  def _compareShapeSparse(self, x_np, use_gpu=False):
    np_ans = np.array(np.shape(x_np))
    x_tf, unused_nnz = _sparsify(x_np)
    with self.cached_session(use_gpu=use_gpu):
      tf_ans = array_ops.shape(x_tf)
      result = self.evaluate(tf_ans)
    self.assertAllEqual(np_ans, result)
    self.assertShapeEqual(np_ans, tf_ans)

  def _compareShapeN(self, x, use_gpu=False):
    np_ans = np.array(np.shape(x))
    with self.cached_session(use_gpu=use_gpu) as sess:
      tf_ans = array_ops.shape_n([x, x, x])
      tf_ans_64 = array_ops.shape_n([x, x, x], out_type=dtypes.int64)
      result = self.evaluate(tf_ans)
      result_64 = self.evaluate(tf_ans_64)
    for i in range(3):
      self.assertAllEqual(np_ans, result[i])
      self.assertAllEqual(np_ans, result_64[i])
      self.assertShapeEqual(np_ans, tf_ans[i])

  def _compareRank(self, x, use_gpu=False):
    np_ans = np.asarray(np.ndim(x))
    with self.cached_session(use_gpu=use_gpu):
      tf_ans = array_ops.rank(x)
      result = self.evaluate(tf_ans)
    self.assertAllEqual(np_ans, result)
    self.assertShapeEqual(np_ans, tf_ans)

  def _compareRankSparse(self, x_np, use_gpu=False):
    np_ans = np.asarray(np.ndim(x_np))
    x_tf, unused_nnz = _sparsify(x_np)
    with self.cached_session(use_gpu=use_gpu):
      tf_ans = array_ops.rank(x_tf)
      result = self.evaluate(tf_ans)
    self.assertAllEqual(np_ans, result)
    self.assertShapeEqual(np_ans, tf_ans)

  def _compareSize(self, x, use_gpu=False):
    np_ans = np.asarray(np.size(x))
    with self.cached_session(use_gpu=use_gpu):
      tf_ans = array_ops.size(x)
      result = self.evaluate(tf_ans)
      tf_ans_64 = array_ops.size(x, out_type=dtypes.int64)
      result_64 = self.evaluate(tf_ans_64)
    self.assertAllEqual(np_ans, result)
    self.assertAllEqual(np_ans, result_64)
    self.assertShapeEqual(np_ans, tf_ans)

  def _compareSizeSparse(self, x_np, use_gpu=False):
    np_ans = np.asarray(np.size(x_np))
    x_tf, unused_nnz = _sparsify(x_np)
    with self.cached_session(use_gpu=use_gpu):
      tf_ans = array_ops.size(x_tf)
      result = self.evaluate(tf_ans)
    self.assertAllEqual(np_ans, result)
    self.assertShapeEqual(np_ans, tf_ans)

  def _compareExpandDims(self, x, dim, use_gpu):
    np_ans = np.expand_dims(x, axis=dim)
    with self.cached_session(use_gpu=use_gpu):
      tensor = array_ops.expand_dims(x, dim)
      tf_ans = self.evaluate(tensor)
    self.assertShapeEqual(np_ans, tensor)
    self.assertAllEqual(np_ans, tf_ans)

  def _compareExpandDimsAll(self, x, dim):
    self._compareExpandDims(x, dim, False)
    self._compareExpandDims(x, dim, True)

  def testExpandDims(self):
    self._compareExpandDimsAll(np.zeros([2]), 0)
    self._compareExpandDimsAll(np.zeros([2]), 1)
    self._compareExpandDimsAll(np.zeros([2]), -1)

    self._compareExpandDimsAll(np.zeros([2, 3]), 0)
    self._compareExpandDimsAll(np.zeros([2, 3]), 1)
    self._compareExpandDimsAll(np.zeros([2, 3]), 2)
    self._compareExpandDimsAll(np.zeros([2, 3]), -1)
    self._compareExpandDimsAll(np.zeros([2, 3]), -2)

    self._compareExpandDimsAll(np.zeros([2, 3, 5]), 0)
    self._compareExpandDimsAll(np.zeros([2, 3, 5]), 1)
    self._compareExpandDimsAll(np.zeros([2, 3, 5]), 2)
    self._compareExpandDimsAll(np.zeros([2, 3, 5]), 3)

    self._compareExpandDimsAll(np.zeros([2, 3, 5]), -1)
    self._compareExpandDimsAll(np.zeros([2, 3, 5]), -2)
    self._compareExpandDimsAll(np.zeros([2, 3, 5]), -3)
    self._compareExpandDimsAll(np.zeros([2, 3, 5]), -4)

  def testExpandDimsBool(self):
    choice = lambda s: np.random.choice((False, True), size=s)
    self._compareExpandDimsAll(choice([2]), 0)
    self._compareExpandDimsAll(choice([2]), 1)
    self._compareExpandDimsAll(choice([2]), -1)

    self._compareExpandDimsAll(choice([2, 3]), 0)
    self._compareExpandDimsAll(choice([2, 3]), 1)
    self._compareExpandDimsAll(choice([2, 3]), 2)
    self._compareExpandDimsAll(choice([2, 3]), -1)
    self._compareExpandDimsAll(choice([2, 3]), -2)

    self._compareExpandDimsAll(choice([2, 3, 5]), 0)
    self._compareExpandDimsAll(choice([2, 3, 5]), 1)
    self._compareExpandDimsAll(choice([2, 3, 5]), 2)
    self._compareExpandDimsAll(choice([2, 3, 5]), 3)

    self._compareExpandDimsAll(choice([2, 3, 5]), -1)
    self._compareExpandDimsAll(choice([2, 3, 5]), -2)
    self._compareExpandDimsAll(choice([2, 3, 5]), -3)
    self._compareExpandDimsAll(choice([2, 3, 5]), -4)

  @test_util.run_deprecated_v1
  def testExpandDimsErrors(self):
    with self.cached_session():
      self.assertRaises(ValueError, array_ops.expand_dims,
                        np.zeros([2, 3, 5]), -5)
      self.assertRaises(ValueError, array_ops.expand_dims,
                        [False, True, True], -5)
      self.assertRaises(ValueError, array_ops.expand_dims,
                        np.zeros([2, 3, 5]), 4)
      self.assertRaises(ValueError, array_ops.expand_dims,
                        [False, True, True], 4)

  @test_util.run_deprecated_v1
  def testExpandDimsGradient(self):
    with self.cached_session():
      inp = constant_op.constant(
          np.random.rand(4, 2).astype("f"), dtype=dtypes.float32)
      squeezed = array_ops.expand_dims(inp, 1)

      err = gradient_checker.compute_gradient_error(inp, [4, 2], squeezed,
                                                    [4, 1, 2])
    self.assertLess(err, 1e-3)

  @test_util.run_deprecated_v1
  def testExpandDimsScalar(self):
    with self.cached_session():
      inp = constant_op.constant(7)
      self.assertAllEqual([7], array_ops.expand_dims(inp, 0))
      self.assertAllEqual([7], array_ops.expand_dims(inp, -1))

      inp = constant_op.constant(True)
      self.assertAllEqual([True], array_ops.expand_dims(inp, 0))
      self.assertAllEqual([True], array_ops.expand_dims(inp, -1))

  def testExpandDimsDimType(self):
    for dtype in [dtypes.int32, dtypes.int64]:
      x = np.zeros([2])
      np_ans = np.expand_dims(x, axis=0)
      with self.cached_session():
        tensor = array_ops.expand_dims(x, constant_op.constant(0, dtype))
        tf_ans = self.evaluate(tensor)
      self.assertShapeEqual(np_ans, tensor)
      self.assertAllEqual(np_ans, tf_ans)

if __name__ == "__main__":
  test.main()
