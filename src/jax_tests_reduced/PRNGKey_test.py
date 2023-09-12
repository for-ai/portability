# Copyright 2018 The JAX Authors.
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


import zlib
from functools import partial
from typing import Any, NamedTuple, Optional, Tuple
from unittest import SkipTest, skipIf

import jax
import numpy as np
import scipy.linalg
import scipy.special
import scipy.stats
from absl.testing import absltest, parameterized
from jax import grad, lax
from jax import numpy as jnp
from jax import prng, random, vmap
from jax._src import core, dtypes
from jax._src import prng as prng_internal
from jax._src import random as jax_random
from jax._src import test_util as jtu
from jax.config import config
from jax.interpreters import xla

from ..utils.timer_wrapper import jax_op_timer, partial_timed

config.parse_flags_with_absl()

float_dtypes = jtu.dtypes.all_floating
complex_dtypes = jtu.dtypes.complex
int_dtypes = jtu.dtypes.all_integer
uint_dtypes = jtu.dtypes.all_unsigned


def _prng_key_as_array(key):
    # TODO(frostig): remove once we upgrade to always enable_custom_prng
    return key.unsafe_raw_array() if config.jax_enable_custom_prng else key


def _maybe_unwrap(key):
    # TODO(frostig): remove once we upgrade to always enable_custom_prng
    unwrap = prng_internal.random_unwrap
    return unwrap(key) if config.jax_enable_custom_prng else key


PRNG_IMPLS = [
    ("threefry2x32", prng.threefry_prng_impl),
    ("rbg", prng.rbg_prng_impl),
    ("unsafe_rbg", prng.unsafe_rbg_prng_impl),
]


class PrngTest(jtu.JaxTestCase):
    def testPRNGValues(self):
        # Test to ensure consistent random values between JAX versions
        timer = jax_op_timer()
        with timer:
            k = random.PRNGKey(0)
            timer.gen.send(k)

        self.assertEqual(
            random.randint(k, (3, 3), 0, 8).dtype, dtypes.canonicalize_dtype(jnp.int_)
        )
        if config.x64_enabled:
            self.assertAllClose(
                random.randint(k, (3, 3), 0, 8, dtype="int64"),
                np.array([[7, 2, 6], [2, 1, 0], [6, 7, 7]], dtype="int64"),
            )
        self.assertAllClose(
            random.randint(k, (3, 3), 0, 8, dtype="int32"),
            np.array([[2, 1, 3], [6, 1, 5], [6, 3, 4]], dtype="int32"),
        )

        self.assertAllClose(
            _prng_key_as_array(random.split(k, 4)),
            np.array(
                [
                    [2285895361, 1501764800],
                    [1518642379, 4090693311],
                    [433833334, 4221794875],
                    [839183663, 3740430601],
                ],
                dtype="uint32",
            ),
        )

        self.assertAllClose(
            _prng_key_as_array(random.fold_in(k, 4)),
            np.array([2285895361, 433833334], dtype="uint32"),
        )
