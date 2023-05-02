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

"""Tests for the LAPAX linear algebra module."""

from functools import partial
import itertools

import numpy as np
import scipy
import scipy.linalg
import scipy as osp

from absl.testing import absltest

import jax
from jax import jit, grad, jvp, vmap
from jax import lax
from jax import numpy as jnp
from jax import scipy as jsp
from jax._src.numpy.util import promote_dtypes_inexact
from jax._src import test_util as jtu

from jax.config import config

config.parse_flags_with_absl()
FLAGS = config.FLAGS

scipy_version = tuple(map(int, scipy.version.version.split(".")[:3]))

T = lambda x: np.swapaxes(x, -1, -2)


float_types = jtu.dtypes.floating
complex_types = jtu.dtypes.complex
int_types = jtu.dtypes.all_integer


class NumpyLinalgTest(jtu.JaxTestCase):
    @jtu.sample_product(
        shape=[(0, 0), (4, 4), (5, 5), (50, 50), (2, 6, 6)],
        dtype=float_types + complex_types,
        compute_left_eigenvectors=[False, True],
        compute_right_eigenvectors=[False, True],
    )
    # TODO(phawkins): enable when there is an eigendecomposition implementation
    # for GPU/TPU.
    def testEig(
        self, shape, dtype, compute_left_eigenvectors, compute_right_eigenvectors
    ):
        rng = jtu.rand_default(self.rng())
        n = shape[-1]
        args_maker = lambda: [rng(shape, dtype)]

        # Norm, adjusted for dimension and type.
        def norm(x):
            norm = np.linalg.norm(x, axis=(-2, -1))
            return norm / ((n + 1) * jnp.finfo(dtype).eps)

        def check_right_eigenvectors(a, w, vr):
            self.assertTrue(np.all(norm(np.matmul(a, vr) - w[..., None, :] * vr) < 100))

        def check_left_eigenvectors(a, w, vl):
            rank = len(a.shape)
            aH = jnp.conj(a.transpose(list(range(rank - 2)) + [rank - 1, rank - 2]))
            wC = jnp.conj(w)
            check_right_eigenvectors(aH, wC, vl)

        (a,) = args_maker()
        results = lax.linalg.eig(
            a,
            compute_left_eigenvectors=compute_left_eigenvectors,
            compute_right_eigenvectors=compute_right_eigenvectors,
        )
        w = results[0]

        if compute_left_eigenvectors:
            check_left_eigenvectors(a, w, results[1])
        if compute_right_eigenvectors:
            check_right_eigenvectors(a, w, results[1 + compute_left_eigenvectors])

        self._CompileAndCheck(partial(jnp.linalg.eig), args_maker, rtol=1e-3)
