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
"""Tests for Python ops defined in sparse_ops."""

import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
import tensorflow.python.ops.sparse_grad  # pylint: disable=unused-import
from tensorflow.python.platform import googletest
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

 
class SparsePlaceholderTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testPlaceholder(self):
    foo = array_ops.sparse_placeholder(dtypes.float32, shape=(10, 47))
    self.assertAllEqual([10, 47], foo.get_shape())
    self.assertAllEqual([None, 2], foo.indices.get_shape().as_list())

  @test_util.run_deprecated_v1
  def testPartialShapePlaceholder(self):
    foo = array_ops.sparse_placeholder(dtypes.float32, shape=(None, 47))
    self.assertAllEqual([None, 47], foo.get_shape().as_list())
    self.assertAllEqual([None, 2], foo.indices.get_shape().as_list())

  @test_util.run_deprecated_v1
  def testNoShapePlaceholder(self):
    foo = array_ops.sparse_placeholder(dtypes.float32, shape=None)
    self.assertAllEqual(None, foo.get_shape())
    self.assertAllEqual([None, None], foo.indices.get_shape().as_list())


if __name__ == "__main__":
  googletest.main()