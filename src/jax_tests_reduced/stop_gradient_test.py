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

import re
import unittest
from functools import partial

import jax
import jax.numpy as jnp  # scan tests use numpy
import jax.scipy as jsp
import numpy as np
from absl.testing import absltest, parameterized
from jax import lax, tree_util
from jax._src import test_util as jtu
from jax.ad_checkpoint import checkpoint
from jax.config import config

from ..utils.timer_wrapper import jax_op_timer, partial_timed

config.parse_flags_with_absl()


def high_precision_dot(a, b):
    return lax.dot(a, b, precision=lax.Precision.HIGHEST)


def posify(matrix):
    return high_precision_dot(matrix, matrix.T.conj())


# Simple optimization routine for testing custom_root
def binary_search(func, x0, low=0.0, high=100.0):
    del x0  # unused

    def cond(state):
        low, high = state
        midpoint = 0.5 * (low + high)
        return (low < midpoint) & (midpoint < high)

    def body(state):
        low, high = state
        midpoint = 0.5 * (low + high)
        update_upper = func(midpoint) > 0
        low = jnp.where(update_upper, low, midpoint)
        high = jnp.where(update_upper, midpoint, high)
        return (low, high)

    solution, _ = lax.while_loop(cond, body, (low, high))
    return solution


# Optimization routine for testing custom_root.
def newton_raphson(func, x0):
    tol = 1e-16
    max_it = 20

    fx0, dfx0 = func(x0), jax.jacobian(func)(x0)
    initial_state = (0, x0, fx0, dfx0)  # (iteration, x, f(x), grad(f)(x))

    def cond(state):
        it, _, fx, _ = state
        return (jnp.max(jnp.abs(fx)) > tol) & (it < max_it)

    def body(state):
        it, x, fx, dfx = state
        step = jnp.linalg.solve(dfx.reshape((-1, fx.size)), fx.ravel()).reshape(
            fx.shape
        )
        x_next = x - step
        fx, dfx = func(x_next), jax.jacobian(func)(x_next)
        return (it + 1, x_next, fx, dfx)

    _, x, _, _ = lax.while_loop(cond, body, initial_state)

    return x


class CustomLinearSolveTest(jtu.JaxTestCase):
    @parameterized.named_parameters(
        {"testcase_name": "nonsymmetric", "symmetric": False},
        {"testcase_name": "symmetric", "symmetric": True},
    )
    def test_custom_linear_solve(self, symmetric):
        def explicit_jacobian_solve(matvec, b):
            test_1 = jnp.linalg.solve(jax.jacobian(matvec)(b), b)
            timer = jax_op_timer()
            with timer:
                result = lax.stop_gradient(test_1)
                timer.gen.send(result)
            return lax.stop_gradient(jnp.linalg.solve(jax.jacobian(matvec)(b), b))

        def matrix_free_solve(matvec, b):
            return lax.custom_linear_solve(
                matvec,
                b,
                explicit_jacobian_solve,
                explicit_jacobian_solve,
                symmetric=symmetric,
            )

        def linear_solve(a, b):
            return matrix_free_solve(partial(high_precision_dot, a), b)

        rng = self.rng()
        a = rng.randn(3, 3)
        if symmetric:
            a = a + a.T
        b = rng.randn(3)
        jtu.check_grads(linear_solve, (a, b), order=2, rtol=3e-3)

        expected = jnp.linalg.solve(a, b)
        actual = jax.jit(linear_solve)(a, b)
        self.assertAllClose(expected, actual)

        c = rng.randn(3, 2)
        expected = jnp.linalg.solve(a, c)
        actual = jax.vmap(linear_solve, (None, 1), 1)(a, c)
        self.assertAllClose(expected, actual)
