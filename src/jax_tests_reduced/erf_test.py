# Copyright 2020 The JAX Authors.
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


import collections
from functools import partial
import itertools
import math
from unittest import SkipTest

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

import jax
from jax import dtypes
from jax import lax
from jax._src import test_util as jtu
from jax.test_util import check_grads

from jax.config import config

config.parse_flags_with_absl()
FLAGS = config.FLAGS


compatible_shapes = [[(3,)], [(), (3, 4), (3, 1), (1, 4)], [(2, 3, 4), (2, 1, 4)]]


GradTestSpec = collections.namedtuple(
    "GradTestSpec", ["op", "nargs", "order", "rng_factory", "dtypes", "name", "tol"]
)


def grad_test_spec(op, nargs, order, rng_factory, dtypes, name=None, tol=None):
    return GradTestSpec(op, nargs, order, rng_factory, dtypes, name or op.__name__, tol)


float_dtypes = jtu.dtypes.all_floating
inexact_dtypes = jtu.dtypes.all_inexact
grad_float_dtypes = jtu.dtypes.floating
grad_complex_dtypes = jtu.dtypes.complex
grad_inexact_dtypes = jtu.dtypes.inexact

LAX_GRAD_OPS = [
    grad_test_spec(
        lax.erf, nargs=1, order=2, rng_factory=jtu.rand_small, dtypes=grad_float_dtypes
    ),
]

GradSpecialValuesTestSpec = collections.namedtuple(
    "GradSpecialValuesTestSpec", ["op", "values", "tol"]
)


def check_grads_bilinear(f, args, order, modes=("fwd", "rev"), atol=None, rtol=None):
    # Can use large eps to make up for numerical inaccuracies since the op is
    # bilinear (relying on the fact that we only check one arg at a time)
    lhs, rhs = args
    check_grads(
        lambda lhs: f(lhs, rhs),
        (lhs,),
        order,
        modes=modes,
        atol=atol,
        rtol=rtol,
        eps=1.0,
    )
    check_grads(
        lambda rhs: f(lhs, rhs),
        (rhs,),
        order,
        modes=modes,
        atol=atol,
        rtol=rtol,
        eps=1.0,
    )


class LaxAutodiffTest(jtu.JaxTestCase):
    @parameterized.parameters(
        itertools.chain.from_iterable(
            jtu.sample_product_testcases(
                [
                    dict(
                        op=rec.op,
                        rng_factory=rec.rng_factory,
                        order=rec.order,
                        tol=rec.tol,
                    )
                ],
                shapes=[
                    shapes
                    for shape_group in compatible_shapes
                    for shapes in itertools.combinations_with_replacement(
                        shape_group, rec.nargs
                    )
                ],
                dtype=rec.dtypes,
            )
            for rec in LAX_GRAD_OPS
        )
    )
    def testOpGrad(self, op, rng_factory, shapes, dtype, order, tol):
        rng = rng_factory(self.rng())
        if jtu.device_under_test() == "tpu":
            if op is lax.pow:
                raise SkipTest("pow grad imprecise on tpu")
            if op is lax.cos:
                order = 1  # 2nd-order gradient is imprecise on TPU.

        tol = (
            jtu.join_tolerance(1.5e-1, tol) if jtu.num_float_bits(dtype) == 32 else tol
        )
        args = tuple(rng(shape, dtype) for shape in shapes)
        check_grads(op, args, order, ["fwd", "rev"], tol, tol)
