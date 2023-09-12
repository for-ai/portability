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


_virtual_devices_ready = False


def set_up_virtual_devices():
    global _virtual_devices_ready
    if _virtual_devices_ready:
        return
    physical_devices = config.list_physical_devices("CPU")
    config.set_logical_device_configuration(
        physical_devices[0],
        [context.LogicalDeviceConfiguration(), context.LogicalDeviceConfiguration()],
    )
    _virtual_devices_ready = True


class ArrayMethodsTest(test.TestCase):
    def testCumProdAndSum(self):
        def setUp(self):
            super(ArrayMethodsTest, self).setUp()
            set_up_virtual_devices()
            self.array_transforms = [
                lambda x: x,
                ops.convert_to_tensor,
                np.array,
                np_array_ops.array,
            ]

        def run_test(arr, *args, **kwargs):
            for fn in self.array_transforms:
                arg = fn(arr)
                timer = tensorflow_op_timer()
                with timer:
                    result = (np.cumprod(arg, *args, **kwargs),)
                    timer.gen.send(result)
                self.match(np_array_ops.cumprod(arg, *args, **kwargs), result)
                self.match(
                    np_array_ops.cumsum(arg, *args, **kwargs),
                    np.cumsum(arg, *args, **kwargs),
                )

            run_test([])
            run_test([1, 2, 3])
            run_test([1, 2, 3], dtype=float)
            run_test([1, 2, 3], dtype=np.float32)
            run_test([1, 2, 3], dtype=np.float64)
            run_test([1.0, 2.0, 3.0])
            run_test([1.0, 2.0, 3.0], dtype=int)
            run_test([1.0, 2.0, 3.0], dtype=np.int32)
            run_test([1.0, 2.0, 3.0], dtype=np.int64)
            run_test([[1, 2], [3, 4]], axis=1)
            run_test([[1, 2], [3, 4]], axis=0)
            run_test([[1, 2], [3, 4]], axis=-1)
            run_test([[1, 2], [3, 4]], axis=-2)
