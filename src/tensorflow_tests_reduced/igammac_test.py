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

"""Tests for special math operations."""

import os

from absl import flags
from absl.testing import parameterized

import numpy as np
import scipy.special as sps

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from ..utils.timer_wrapper import tensorflow_op_timer


NUM_SAMPLES = int(1e3)


@def_function.function(jit_compile=True)
def _igamma(a, x):
    return math_ops.igamma(a, x)


@def_function.function(jit_compile=True)
def _igammac(a, x):
    timer = tensorflow_op_timer()
    with timer:
        result = math_ops.igammac(a, x)
        timer.gen.send(result)
    return result


@def_function.function(jit_compile=True)
def _polygamma(n, x):
    return math_ops.polygamma(n, x)


@def_function.function(jit_compile=True)
def _zeta(a, q):
    return math_ops.zeta(a, q)


# This is df/da / df/dx, where f = igamma.
def implicit_reparameterization_grad(a, x):
    log_prob = math_ops.xlogy(a - 1.0, x) - math_ops.lgamma(a) - x
    prob = math_ops.exp(log_prob)
    return -gen_math_ops.igamma_grad_a(a, x) / prob


@def_function.function(jit_compile=True)
def _log1p(x):
    return math_ops.log1p(x)


class IgammacTest(test.TestCase, parameterized.TestCase):
    @parameterized.parameters((np.float32, 1e-2, 1e-11), (np.float64, 1e-4, 1e-30))
    def testLargeXSmallA(self, dtype, rtol, atol):
        # rtol, atol = self.adjust_tolerance_for_tpu(dtype, rtol, atol)
        # Test values near zero.
        x = np.random.uniform(low=100.0, high=200.0, size=[NUM_SAMPLES]).astype(dtype)
        a = np.random.uniform(low=0.3, high=1.0, size=[NUM_SAMPLES]).astype(dtype)

        expected_values = sps.gammaincc(a, x)
        actual = _igammac(a, x)
        self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)

    @parameterized.parameters((np.float32, 1e-2, 1e-11), (np.float64, 1e-4, 1e-30))
    def testSmallValues(self, dtype, rtol, atol):
        # rtol, atol = self.adjust_tolerance_for_tpu(dtype, rtol, atol)
        # Test values near zero.
        x = np.random.uniform(
            low=np.finfo(dtype).tiny, high=1.0, size=[NUM_SAMPLES]
        ).astype(dtype)
        a = np.random.uniform(
            low=np.finfo(dtype).tiny, high=1.0, size=[NUM_SAMPLES]
        ).astype(dtype)

        expected_values = sps.gammaincc(a, x)
        actual = _igammac(a, x)
        self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)
