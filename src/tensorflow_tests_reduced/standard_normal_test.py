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
"""Tests for tf numpy random number methods."""
# pylint: disable=g-direct-tensorflow-import

from absl.testing import parameterized
import numpy as onp

from tensorflow.python.framework import ops
from tensorflow.python.ops import numpy_ops as np
# Needed for ndarray.reshape.
from tensorflow.python.ops.numpy_ops import np_array_ops  # pylint: disable=unused-import
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_math_ops
from tensorflow.python.ops.numpy_ops import np_random
from tensorflow.python.platform import test
from ..utils.timer_wrapper import tensorflow_op_timer


class RandomTestBase(test.TestCase, parameterized.TestCase):

    def _test(self, *args, **kw_args):
        onp_dtype = kw_args.pop('onp_dtype', None)
        allow_float64 = kw_args.pop('allow_float64', True)
        old_allow_float64 = np_dtypes.is_allow_float64()
        np_dtypes.set_allow_float64(allow_float64)
        old_func = getattr(self, 'onp_func', None)
        # TODO(agarwal): Note that onp can return a scalar type while np returns
        # ndarrays. Currently np does not support scalar types.
        self.onp_func = lambda *args, **kwargs: onp.asarray(  # pylint: disable=g-long-lambda
            old_func(*args, **kwargs))
        np_out = self.np_func(*args, **kw_args)
        onp_out = onp.asarray(self.onp_func(*args, **kw_args))
        if onp_dtype is not None:
            onp_out = onp_out.astype(onp_dtype)
        self.assertEqual(np_out.shape, onp_out.shape)
        self.assertEqual(np_out.dtype, onp_out.dtype)
        np_dtypes.set_allow_float64(old_allow_float64)


class StandardNormalTest(RandomTestBase):

    def setUp(self):
        self.np_func = np.random.standard_normal
        timer = tensorflow_op_timer()
        with timer:
            self.onp_func = onp.random.standard_normal
            timer.gen.send(self)
        super(StandardNormalTest, self).setUp()

    @parameterized.parameters((None,), ((),), ((1,),), ((1, 2),))
    def test(self, size):
        self._test(size)


if __name__ == '__main__':
    ops.enable_eager_execution()
    np_math_ops.enable_numpy_methods_on_tensor()
    test.main()
