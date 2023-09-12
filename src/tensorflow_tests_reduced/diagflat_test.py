# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf numpy array methods."""

import itertools
import operator
import sys
from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_math_ops
from tensorflow.python.platform import test
from ..utils.timer_wrapper import tensorflow_op_timer
from tensorflow.python.framework import test_util


_virtual_devices_ready = False


# def set_up_virtual_devices():
#     global _virtual_devices_ready
#     print("***DEVICES READY?", _virtual_devices_ready)
#     if _virtual_devices_ready:
#         return
#     physical_devices = config.list_physical_devices('CPU')
#     config.set_logical_device_configuration(
#         physical_devices[0], [
#             context.LogicalDeviceConfiguration(),
#             context.LogicalDeviceConfiguration()
#         ])
#     _virtual_devices_ready = True


class ArrayCreationTest(test.TestCase):
    def setUp(self):
        np_math_ops.enable_numpy_methods_on_tensor()
        super(ArrayCreationTest, self).setUp()
        # set_up_virtual_devices()
        python_shapes = [
            0, 1, 2, (), (1,), (2,), (1, 2, 3), [], [1], [2], [1, 2, 3]
        ]
        self.shape_transforms = [
            lambda x: x, lambda x: np.array(x, dtype=int),
            lambda x: np_array_ops.array(
                x, dtype=int), tensor_shape.TensorShape
        ]

        self.all_shapes = []
        for fn in self.shape_transforms:
            self.all_shapes.extend([fn(s) for s in python_shapes])

        if sys.version_info.major == 3:
            # There is a bug of np.empty (and alike) in Python 3 causing a crash when
            # the `shape` argument is an np_arrays.ndarray scalar (or tf.Tensor
            # scalar).
            def not_ndarray_scalar(s):
                return not (isinstance(s, np_arrays.ndarray) and s.ndim == 0)

            self.all_shapes = list(filter(not_ndarray_scalar, self.all_shapes))

        self.all_types = [
            int, float, np.int16, np.int32, np.int64, np.float16, np.float32,
            np.float64, np.complex64, np.complex128
        ]

        source_array_data = [
            1,
            5.5,
            7,
            (),
            (8, 10.),
            ((), ()),
            ((1, 4), (2, 8)),
            [],
            [7],
            [8, 10.],
            [[], []],
            [[1, 4], [2, 8]],
            ([], []),
            ([1, 4], [2, 8]),
            [(), ()],
            [(1, 4), (2, 8)],
        ]

        self.array_transforms = [
            lambda x: x,
            ops.convert_to_tensor,
            np.array,
            np_array_ops.array,
        ]
        self.all_arrays = []
        for fn in self.array_transforms:
            self.all_arrays.extend([fn(s) for s in source_array_data])

    def match_shape(self, actual, expected, msg=None):
        if msg:
            msg = 'Shape match failed for: {}. Expected: {} Actual: {}'.format(
                msg, expected.shape, actual.shape)
        self.assertEqual(actual.shape, expected.shape, msg=msg)

    def match_dtype(self, actual, expected, msg=None):
        if msg:
            msg = 'Dtype match failed for: {}. Expected: {} Actual: {}.'.format(
                msg, expected.dtype, actual.dtype)
        self.assertEqual(actual.dtype, expected.dtype, msg=msg)

    def match(self, actual, expected, msg=None, check_dtype=True):
        msg_ = 'Expected: {} Actual: {}'.format(expected, actual)
        if msg:
            msg = '{} {}'.format(msg_, msg)
        else:
            msg = msg_
        self.assertIsInstance(actual, np_arrays.ndarray)
        if check_dtype:
            self.match_dtype(actual, expected, msg)
        self.match_shape(actual, expected, msg)
        if not actual.shape.rank:
            self.assertAllClose(actual.tolist(), expected.tolist())
        else:
            self.assertAllClose(actual.tolist(), expected.tolist())

    def testDiagFlat(self):
        array_transforms = [
            lambda x: x,  # Identity,
            ops.convert_to_tensor,
            np.array,
            lambda x: np.array(x, dtype=np.float32),
            lambda x: np.array(x, dtype=np.float64),
            np_array_ops.array,
            lambda x: np_array_ops.array(x, dtype=np.float32),
            lambda x: np_array_ops.array(x, dtype=np.float64)
        ]

        def run_test(arr):
            for fn in array_transforms:
                import tensorflow as tf
                with tf.device('cpu'):
                    arr = fn(arr)
                timer = tensorflow_op_timer()
                with timer:
                    test_2 = np_array_ops.diagflat(arr)
                    timer.gen.send(test_2)
                self.match(
                    np_array_ops.diagflat(arr),
                    np.diagflat(arr),
                    msg='diagflat({})'.format(arr))
                for k in range(-3, 3):
                    timer = tensorflow_op_timer()
                    with timer:
                        test_1 = np_array_ops.diagflat(arr, k)
                        timer.gen.send(test_1)
                    self.match(
                        np_array_ops.diagflat(arr, k),
                        np.diagflat(arr, k),
                        msg='diagflat({}, k={})'.format(arr, k))
        # 1-d arrays.
        run_test([])
        run_test([1])
        run_test([1, 2])
        # 2-d arrays.
        run_test([[]])
        run_test([[5]])
        run_test([[], []])
        run_test(np.arange(4).reshape((2, 2)).tolist())
        run_test(np.arange(2).reshape((2, 1)).tolist())
        run_test(np.arange(2).reshape((1, 2)).tolist())
        # 3-d arrays
        run_test(np.arange(8).reshape((2, 2, 2)).tolist())


if __name__ == '__main__':
    test.main()
