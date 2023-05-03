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


import enum
from functools import partial
import itertools
import typing
from typing import Any, Optional, Tuple
import warnings

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

import jax
from jax import lax
from jax import numpy as jnp
from jax import ops

from jax._src import dtypes
from jax._src import test_util as jtu
from jax._src import util
from jax._src.lax import lax as lax_internal

from jax.config import config
from ..utils.timer_wrapper import jax_op_timer

config.parse_flags_with_absl()
ARRAY_MSG = r"Using a non-tuple sequence for multidimensional indexing is not allowed.*arr\[array\(seq\)\]"
TUPLE_MSG = r"Using a non-tuple sequence for multidimensional indexing is not allowed.*arr\[tuple\(seq\)\]"


class IndexedUpdateTest(jtu.JaxTestCase):
    @jtu.sample_product(
        [
            dict(idx=idx, idx_type=idx_type)
            for idx, idx_type in [
                ([0], "array"),
                ([0, 0], "array"),
                ([[0, 0]], "tuple"),
                ([0, [0, 1]], "tuple"),
                ([0, np.arange(2)], "tuple"),
                ([0, None], "tuple"),
                ([0, slice(None)], "tuple"),
            ]
        ],
    )
    def testIndexSequenceDeprecation(self, idx, idx_type):
        normalize = {"array": np.array, "tuple": tuple}[idx_type]
        msg = {"array": ARRAY_MSG, "tuple": TUPLE_MSG}[idx_type]
        x = jnp.arange(6)
        timer = jax_op_timer()
        with timer:
            x = x.reshape(3, 2)
            timer.gen.send(x)

        with self.assertRaisesRegex(TypeError, msg):
            x[idx]
        with self.assertNoWarnings():
            x[normalize(idx)]

        with self.assertRaisesRegex(TypeError, msg):
            x.at[idx].set(0)
        with self.assertNoWarnings():
            x.at[normalize(idx)].set(0)
