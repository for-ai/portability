"""Tests for Python ops defined in sparse_ops."""

import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import googletest
from tensorflow.python.framework import sparse_tensor


class SparseFillEmptyRowsTest(test_util.TensorFlowTestCase):
    def _SparseTensorValue_5x6(self, dtype=np.int32):
        ind = np.array([[0, 0], [1, 0], [1, 3], [1, 4], [3, 2], [3, 3]])
        val = np.array([0, 10, 13, 14, 32, 33])
        shape = np.array([5, 6])
        return sparse_tensor.SparseTensorValue(
            np.array(ind, np.int64), np.array(val, dtype), np.array(
                shape, np.int64))

    def _SparseTensor_5x6(self):
        return sparse_tensor.SparseTensor.from_value(self._SparseTensorValue_5x6())

    def testFillNumber(self):
        with test_util.use_gpu():
            for sp_input in (self._SparseTensorValue_5x6(), self._SparseTensor_5x6()):
                sp_output, empty_row_indicator = (
                    sparse_ops.sparse_fill_empty_rows(sp_input, -1))

                output, empty_row_indicator_out = self.evaluate(
                    [sp_output, empty_row_indicator])

                self.assertAllEqual(
                    output.indices,
                    [[0, 0], [1, 0], [1, 3], [1, 4], [2, 0], [3, 2], [3, 3], [4, 0]])
                self.assertAllEqual(
                    output.values, [0, 10, 13, 14, -1, 32, 33, -1])
                self.assertAllEqual(output.dense_shape, [5, 6])
                self.assertAllEqual(empty_row_indicator_out,
                                    np.array([0, 0, 1, 0, 1]).astype(np.bool_))


if __name__ == "__main__":
    googletest.main()
