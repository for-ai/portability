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
from __future__ import annotations

from functools import partial
import itertools
import math
import operator
import types
import unittest
from unittest import SkipTest
from typing import Tuple

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

import jax
from jax._src import core
from jax import lax
import jax.numpy as jnp
from jax.test_util import check_grads
from jax import tree_util
import jax.util

from jax.interpreters import xla
from jax._src.interpreters import mlir
from jax.interpreters import batching
from jax._src import array
from jax._src.lib.mlir.dialects import hlo
from jax._src import dtypes
from jax._src.interpreters import pxla
from jax._src import test_util as jtu
from jax._src import lax_reference
from jax._src.lax import lax as lax_internal
from jax._src.internal_test_util import lax_test_util

from jax.config import config

config.parse_flags_with_absl()


### lax tests

# We check cases where the preferred type is at least as wide as the input
# type and where both are either both floating-point or both integral,
# which are the only supported configurations.
preferred_type_combinations = [
    (np.float16, np.float16),
    (np.float16, np.float32),
    (np.float16, np.float64),
    (dtypes.bfloat16, dtypes.bfloat16),
    (dtypes.bfloat16, np.float32),
    (dtypes.bfloat16, np.float64),
    (np.float32, np.float32),
    (np.float32, np.float64),
    (np.float64, np.float64),
    (np.int8, np.int8),
    (np.int8, np.int16),
    (np.int8, np.int32),
    (np.int8, np.int64),
    (np.int16, np.int16),
    (np.int16, np.int32),
    (np.int16, np.int64),
    (np.int32, np.int32),
    (np.int32, np.int64),
    (np.int64, np.int64),
    (np.complex64, np.complex64),
    (np.complex64, np.complex128),
    (np.complex128, np.complex128),
    (np.int8, np.float16),
    (np.int8, dtypes.bfloat16),
    (np.int8, np.float32),
    (np.int8, np.float64),
    (np.int16, np.float16),
    (np.int16, dtypes.bfloat16),
    (np.int16, np.float32),
    (np.int16, np.float64),
    (np.int32, np.float32),
    (np.int32, np.float64),
    (np.int64, np.float64),
]


class LaxTest(jtu.JaxTestCase):
    """Numerical tests for LAX operations."""

    @jtu.sample_product(
        dtype=[np.float32, np.int32, np.uint32],
        shape=[(3,), (5, 3)],
        k=[1, 3],
    )
    def testTopK(self, shape, dtype, k):
        def args_maker():
            flat_values = np.arange(math.prod(shape), dtype=dtype)
            values = self.rng().permutation(flat_values).reshape(shape)
            return [values]

        def reference_top_k(x):
            bcast_idxs = np.broadcast_to(np.arange(shape[-1], dtype=np.int32), shape)
            sorted_vals, sorted_idxs = lax_reference.sort_key_val(x, bcast_idxs)
            return sorted_vals[..., : -k - 1 : -1], sorted_idxs[..., : -k - 1 : -1]

        op = lambda vs: lax.top_k(vs, k=k)
        self._CheckAgainstNumpy(op, reference_top_k, args_maker)
        self._CompileAndCheck(op, args_maker)
