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
import itertools
import typing
import warnings
from functools import partial
from typing import Any, Optional, Tuple

import jax
import numpy as np
from absl.testing import absltest, parameterized
from jax import lax
from jax import numpy as jnp
from jax import ops
from jax._src import dtypes
from jax._src import test_util as jtu
from jax._src import util
from jax._src.lax import lax as lax_internal
from jax.config import config

from ..utils.timer_wrapper import jax_op_timer, partial_timed

config.parse_flags_with_absl()

# We disable the whitespace continuation check in this file because otherwise it
# makes the test name formatting unwieldy.
# pylint: disable=bad-continuation


ARRAY_MSG = r"Using a non-tuple sequence for multidimensional indexing is not allowed.*arr\[array\(seq\)\]"
TUPLE_MSG = r"Using a non-tuple sequence for multidimensional indexing is not allowed.*arr\[tuple\(seq\)\]"


float_dtypes = jtu.dtypes.floating
default_dtypes = float_dtypes + jtu.dtypes.integer
all_dtypes = default_dtypes + jtu.dtypes.boolean


class IndexingTest(jtu.JaxTestCase):
    """Tests for Numpy indexing translation rules."""

    @jtu.sample_product(
        funcname=["log"],
    )
    def testIndexApply(self, funcname, size=10, dtype="float32"):
        rng = jtu.rand_default(self.rng())
        idx_rng = jtu.rand_int(self.rng(), -size, size)
        np_func = getattr(np, funcname)
        timer = jax_op_timer()
        with timer:
            jnp_func = getattr(jnp, funcname)
            timer.gen.send(jnp_func)

        @jtu.ignore_warning(category=RuntimeWarning)
        def np_op(x, idx):
            y = x.copy()
            np_func.at(y, idx)
            return y

        def jnp_op(x, idx):
            return jnp.asarray(x).at[idx].apply(jnp_func)

        args_maker = lambda: [rng(size, dtype), idx_rng(size, int)]
        self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
        self._CompileAndCheck(jnp_op, args_maker)
