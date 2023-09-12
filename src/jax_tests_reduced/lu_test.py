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

import itertools
from functools import partial

import jax
import numpy as np
import scipy
import scipy as osp
import scipy.linalg
from absl.testing import absltest
from jax import grad, jit, jvp, lax
from jax import numpy as jnp
from jax import scipy as jsp
from jax import vmap
from jax._src import test_util as jtu
from jax._src.numpy.util import promote_dtypes_inexact
from jax.config import config

from ..utils.timer_wrapper import jax_op_timer, partial_timed

config.parse_flags_with_absl()
FLAGS = config.FLAGS

scipy_version = tuple(map(int, scipy.version.version.split(".")[:3]))

T = lambda x: np.swapaxes(x, -1, -2)


float_types = jtu.dtypes.floating
complex_types = jtu.dtypes.complex
int_types = jtu.dtypes.all_integer


class ScipyLinalgTest(jtu.JaxTestCase):
    @jtu.sample_product(
        shape=[(1, 1), (4, 5), (10, 5), (50, 50)],
        dtype=float_types + complex_types,
    )
    def testLu(self, shape, dtype):
        rng = jtu.rand_default(self.rng())
        args_maker = lambda: [rng(shape, dtype)]
        (x,) = args_maker()
        timer = jax_op_timer()
        with timer:
            p, l, u = jsp.linalg.lu(x)
            timer.gen.send((p, l, u))

        self.assertAllClose(
            x,
            np.matmul(p, np.matmul(l, u)),
            rtol={
                np.float32: 1e-3,
                np.float64: 1e-12,
                np.complex64: 1e-3,
                np.complex128: 1e-12,
            },
            atol={np.float32: 1e-5},
        )
        self._CompileAndCheck(jsp.linalg.lu, args_maker)

    def testLuOfSingularMatrix(self):
        x = jnp.array([[-1.0, 3.0 / 2], [2.0 / 3, -1.0]], dtype=np.float32)
        timer = jax_op_timer()
        with timer:
            p, l, u = jsp.linalg.lu(x)
            timer.gen.send((p, l, u))
        self.assertAllClose(x, np.matmul(p, np.matmul(l, u)))

    @jtu.sample_product(
        shape=[(1, 1), (4, 5), (10, 5), (10, 10), (6, 7, 7)],
        dtype=float_types + complex_types,
    )
    def testLuGrad(self, shape, dtype):
        rng = jtu.rand_default(self.rng())
        a = rng(shape, dtype)
        lu = vmap(jsp.linalg.lu) if len(shape) > 2 else jsp.linalg.lu
        jtu.check_grads(lu, (a,), 2, atol=5e-2, rtol=3e-1)

    @jtu.sample_product(
        shape=[(4, 5), (6, 5)],
        dtype=[jnp.float32],
    )
    def testLuBatching(self, shape, dtype):
        rng = jtu.rand_default(self.rng())
        args = [rng(shape, jnp.float32) for _ in range(10)]
        for x in args: 
            timer = jax_op_timer()
            with timer:
                result = osp.linalg.lu(x)
                timer.gen.send(result)
        expected = list(osp.linalg.lu(x) for x in args)
        ps = np.stack([out[0] for out in expected])
        ls = np.stack([out[1] for out in expected])
        us = np.stack([out[2] for out in expected])

        actual_ps, actual_ls, actual_us = vmap(jsp.linalg.lu)(jnp.stack(args))
        self.assertAllClose(ps, actual_ps)
        self.assertAllClose(ls, actual_ls, rtol=5e-6)
        self.assertAllClose(us, actual_us)

    @jtu.skip_on_devices("cpu", "tpu")
    def testLuCPUBackendOnGPU(self):
        # tests running `lu` on cpu when a gpu is present.
        jit(jsp.linalg.lu, backend="cpu")(np.ones((2, 2)))  # does not crash
