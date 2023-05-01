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


from collections import defaultdict
from functools import partial
import itertools

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import lax
import jax.numpy as jnp
import jax._src.test_util as jtu

from jax.config import config

config.parse_flags_with_absl()


class EinsumTest(jtu.JaxTestCase):
    def test_einsum_path(self):
        # just check examples from np.einsum_path docstring
        a = self.rng().rand(2, 2)
        b = self.rng().rand(2, 5)
        c = self.rng().rand(5, 2)

        path_info = np.einsum_path("ij,jk,kl->il", a, b, c, optimize="greedy")
        self.assertEqual(str(path_info[0]), "['einsum_path', (1, 2), (0, 1)]")
        self.assertEqual(
            path_info[1].split("\n")[0], "  Complete contraction:  ij,jk,kl->il"
        )

        # check this doesn't crash
        I = self.rng().rand(10, 10, 10, 10)
        C = self.rng().rand(10, 10)
        np.einsum_path("ea,fb,abcd,gc,hd->efgh", C, C, I, C, C, optimize="greedy")

    def test_einsum_kpmurphy_example(self):
        # code from an email with @murphyk
        N, C, D, K, T = 2, 3, 4, 5, 6
        r = self.rng()
        S = r.randn(N, T, K)
        W = r.randn(K, D)
        V = r.randn(D, C)
        L = np.zeros((N, C))
        for n in range(N):
            for c in range(C):
                s = 0
                for d in range(D):
                    for k in range(K):
                        for t in range(T):
                            s += S[n, t, k] * W[k, d] * V[d, c]
                L[n, c] = s

        path = jnp.einsum_path("ntk,kd,dc->nc", S, W, V, optimize="optimal")[0]
        rtol = 1e-2 if jtu.device_under_test() == "tpu" else None
        self.assertAllClose(
            L,
            jnp.einsum("ntk,kd,dc->nc", S, W, V, optimize=path),
            check_dtypes=False,
            rtol=rtol,
        )
