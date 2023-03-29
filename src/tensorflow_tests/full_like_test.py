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


_virtual_devices_ready = False


def set_up_virtual_devices():
    global _virtual_devices_ready
    if _virtual_devices_ready:
        return
    physical_devices = config.list_physical_devices('CPU')
    config.set_logical_device_configuration(
        physical_devices[0], [
            context.LogicalDeviceConfiguration(),
            context.LogicalDeviceConfiguration()
        ])
    _virtual_devices_ready = True


class ArrayCreationTest(test.TestCase):

    def setUp(self):
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

    def match(self, actual, expected, msg=None, almost=False, decimal=7):
        msg_ = 'Expected: {} Actual: {}'.format(expected, actual)
        if msg:
            msg = '{} {}'.format(msg_, msg)
        else:
            msg = msg_
        self.assertIsInstance(actual, np_arrays.ndarray)
        self.match_dtype(actual, expected, msg)
        self.match_shape(actual, expected, msg)
        if not almost:
            if not actual.shape.rank:
                self.assertEqual(actual.tolist(), expected.tolist())
            else:
                self.assertSequenceEqual(actual.tolist(), expected.tolist())
        else:
            np.testing.assert_almost_equal(
                actual.tolist(), expected.tolist(), decimal=decimal)

    def testFullLike(self):
        # List of 2-tuples of fill value and shape.
        data = [
            (5, ()),
            (5, (7,)),
            (5., (7,)),
            ([5, 8], (2,)),
            ([5, 8], (3, 2)),
            ([[5], [8]], (2, 3)),
            ([[5], [8]], (3, 2, 5)),
            ([[5.], [8.]], (3, 2, 5)),
        ]
        zeros_builders = [np_array_ops.zeros, np.zeros]
        for f, s in data:
            for fn1, fn2, arr_dtype in itertools.product(self.array_transforms,
                                                         zeros_builders,
                                                         self.all_types):
                fill_value = fn1(f)
                arr = fn2(s, arr_dtype)
                self.match(
                    np_array_ops.full_like(arr, fill_value),
                    np.full_like(arr, fill_value))
                for dtype in self.all_types:
                    self.match(
                        np_array_ops.full_like(arr, fill_value, dtype=dtype),
                        np.full_like(arr, fill_value, dtype=dtype))


ops.enable_eager_execution()
ops.enable_numpy_style_type_promotion()
np_math_ops.enable_numpy_methods_on_tensor()

if __name__ == '__main__':
    test.main()
