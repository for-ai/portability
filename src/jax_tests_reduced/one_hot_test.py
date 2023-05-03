# Copyright 2019 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for nn module."""

import collections
import itertools
from functools import partial

import jax
import jax.numpy as jnp
import scipy.stats
from absl.testing import absltest, parameterized
from jax import nn, random
from jax._src import core
from jax._src import test_util as jtu
from jax.config import config
from jax.test_util import check_grads

from ..utils.timer_wrapper import jax_op_timer, partial_timed

config.parse_flags_with_absl()


class NNFunctionsTest(jtu.JaxTestCase):
    def testOneHot(self):
        timer = jax_op_timer()
        with timer:
            actual = nn.one_hot(jnp.array([0, 1, 2]), 3)
            timer.gen.send(actual)
        expected = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertAllClose(actual, expected, check_dtypes=False)
        timer = jax_op_timer()
        with timer:
            actual = nn.one_hot(jnp.array([1, 2, 0]), 3)
            timer.gen.send(actual)
        expected = jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        self.assertAllClose(actual, expected, check_dtypes=False)
