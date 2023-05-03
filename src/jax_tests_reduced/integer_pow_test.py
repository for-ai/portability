# Copyright 2021 The JAX Authors.
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

from functools import partial
import operator

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

import jax
from jax import config, jit, lax
import jax.numpy as jnp
import jax._src.test_util as jtu
from jax.experimental.sparse import BCOO, sparsify, todense, SparseTracer
from jax.experimental.sparse.transform import (
    arrays_to_spvalues,
    spvalues_to_arrays,
    sparsify_raw,
    SparsifyValue,
    SparsifyEnv,
)
from jax.experimental.sparse.util import CuSparseEfficiencyWarning
from jax.experimental.sparse import test_util as sptu

config.parse_flags_with_absl()


def rand_sparse(rng, nse=0.5, post=lambda x: x, rand_method=jtu.rand_default):
    def _rand_sparse(shape, dtype, nse=nse):
        rand = rand_method(rng)
        size = np.prod(shape).astype(int)
        if 0 <= nse < 1:
            nse = nse * size
        nse = min(size, int(nse))
        M = rand(shape, dtype)
        indices = rng.choice(size, size - nse, replace=False)
        M.flat[indices] = 0
        return post(M)

    return _rand_sparse


class SparsifyTest(jtu.JaxTestCase):
    @classmethod
    def sparsify(cls, f):
        return sparsify(f, use_tracer=False)

    @parameterized.named_parameters(
        {
            "testcase_name": f"_{op.__name__}_{fmt}",
            "op": op,
            "dtype": dtype,
            "kwds": kwds,
            "fmt": fmt,
        }
        for fmt in ["BCSR", "BCOO"]
        for op, dtype, kwds in [
            (lax.integer_pow, jnp.float32, {"y": 2}),
        ]
    )
    def testUnaryOperationsNonUniqueIndices(self, fmt, op, dtype, kwds):
        shape = (4, 5)

        # Note: we deliberately test non-unique indices here.
        if fmt == "BCOO":
            rng = sptu.rand_bcoo(self.rng())
        elif fmt == "BCSR":
            rng = sptu.rand_bcsr(self.rng())
        else:
            raise ValueError(f"Unrecognized {fmt=}")
        mat = rng(shape, dtype)

        sparse_result = self.sparsify(partial(op, **kwds))(mat)
        dense_result = op(mat.todense(), **kwds)

        self.assertArraysAllClose(sparse_result.todense(), dense_result)

        # Ops that commute with addition should not deduplicate indices.
        if op in [jnp.copy, lax.neg, lax.real, lax.imag]:
            self.assertArraysAllClose(sparse_result.indices, mat.indices)
            if fmt == "BCSR":
                self.assertArraysAllClose(sparse_result.indptr, mat.indptr)
