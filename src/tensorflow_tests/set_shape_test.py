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

class SparseSetShapeTest(test_util.TensorFlowTestCase):

  def testSetShapeEagerValidates(self):
    ind = np.array([[0, 0], [1, 0], [1, 3], [1, 4], [3, 2], [3, 3]])
    val = np.array([0, 10, 13, 14, 32, 33])
    shape = np.array([5, 6])
    sp = sparse_tensor.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtypes.int64),
        constant_op.constant(shape, dtypes.int64))

    self.assertEqual(sp.shape, tensor_shape.TensorShape([5, 6]))

    sp.set_shape(tensor_shape.TensorShape(None))
    sp.set_shape(tensor_shape.TensorShape([None, None]))
    sp.set_shape(tensor_shape.TensorShape([5, None]))
    sp.set_shape(tensor_shape.TensorShape([None, 6]))
    sp.set_shape(tensor_shape.TensorShape([5, 6]))

    with self.assertRaises(ValueError):
      sp.set_shape([None, None, None])

    with self.assertRaises(ValueError):
      sp.set_shape([3, None])

    with self.assertRaises(ValueError):
      sp.set_shape([None, 7])

    with self.assertRaises(ValueError):
      sp.set_shape([3, 6])

  def testSetShapeFunctionMerges(self):

    @def_function.function
    def dynamic_shape_sparse(dense_shape):
      ind = np.array([[0, 0], [1, 0], [1, 3], [1, 4], [3, 2], [3, 3]])
      val = np.array([0, 10, 13, 14, 32, 33])
      sp = sparse_tensor.SparseTensor(
          constant_op.constant(ind, dtypes.int64),
          constant_op.constant(val, dtypes.int64),
          dense_shape)

      sp.set_shape(tensor_shape.TensorShape(None))
      self.assertEqual(sp.shape, tensor_shape.TensorShape(None))

      sp.set_shape(tensor_shape.TensorShape([None, None]))
      self.assertEqual(sp.shape, tensor_shape.TensorShape([None, None]))

      sp.set_shape(tensor_shape.TensorShape([5, None]))
      self.assertEqual(sp.shape, tensor_shape.TensorShape([5, None]))

      sp.set_shape(tensor_shape.TensorShape([None, 6]))
      self.assertEqual(sp.shape, tensor_shape.TensorShape([5, 6]))

      sp.set_shape(tensor_shape.TensorShape([None, None]))
      self.assertEqual(sp.shape, tensor_shape.TensorShape([5, 6]))

      sp.set_shape(tensor_shape.TensorShape([5, 6]))
      self.assertEqual(sp.shape, tensor_shape.TensorShape([5, 6]))

      with self.assertRaises(ValueError):
        sp.set_shape([None, None, None])

      with self.assertRaises(ValueError):
        sp.set_shape([3, None])

      with self.assertRaises(ValueError):
        sp.set_shape([None, 7])

      with self.assertRaises(ValueError):
        sp.set_shape([3, 6])

    dense_shape_spec = tensor_spec.TensorSpec(None, dtypes.int64)
    _ = dynamic_shape_sparse.get_concrete_function(dense_shape_spec)



if __name__ == "__main__":
  googletest.main()
