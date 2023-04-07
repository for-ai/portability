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
"""Tests for batch_norm related functionality in tensorflow.ops.nn."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class NormalizeMomentsTest(test.TestCase):

    def _npNormalizeMoments(self, counts, mean_ss, variance_ss, shift):
        mean = mean_ss / counts
        variance = variance_ss / counts - mean * mean
        if shift is not None:
            mean += shift
        return mean, variance

    def _opNormalizeMoments(self, counts, mean_ss, variance_ss, shift):
        return nn_impl.normalize_moments(counts, mean_ss, variance_ss, shift)

    def _testNormalizeMoments(self, shape, shift):
        counts = np.ones([1]).astype(np.float32)
        mean_ss = np.random.random_sample(shape).astype(np.float32)
        variance_ss = np.random.random_sample(shape).astype(np.float32)
        variance_ss *= variance_ss
        if shift:
            shift_v = np.random.random_sample(shape).astype(np.float32)
        else:
            shift_v = None
        npm, npv = self._npNormalizeMoments(
            counts, mean_ss, variance_ss, shift_v)
        for use_gpu in [True, False]:
            with self.cached_session(use_gpu=use_gpu) as sess:
                tf_counts = constant_op.constant(counts, name="counts")
                tf_mean_ss = constant_op.constant(mean_ss, name="mean_ss")
                tf_variance_ss = constant_op.constant(
                    variance_ss, name="variance_ss")
                if shift:
                    tf_shift_v = constant_op.constant(shift_v, name="shift")
                else:
                    tf_shift_v = None
                opm, opv = self._opNormalizeMoments(tf_counts, tf_mean_ss,
                                                    tf_variance_ss, tf_shift_v)
                tfm, tfv = self.evaluate([opm, opv])
                self.assertAllClose(npm, tfm, atol=0.000001)
                self.assertAllClose(npv, tfv, atol=0.000001)

    def testNormalizeMoments(self):
        for shift in [None, 4.0]:
            self._testNormalizeMoments([3], shift)
            self._testNormalizeMoments([2, 3], shift)


if __name__ == "__main__":
    test.main()
