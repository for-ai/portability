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

import collections
import functools
import itertools

import jax
import numpy as np
import scipy.special as osp_special
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.config import config
from jax.scipy import special as lsp_special

config.parse_flags_with_absl()
FLAGS = config.FLAGS
from ..utils.timer_wrapper import jax_op_timer, partial_timed

all_shapes = [(), (4,), (3, 4), (3, 1), (1, 4), (2, 1, 4)]

OpRecord = collections.namedtuple(
    "OpRecord",
    [
        "name",
        "nargs",
        "dtypes",
        "rng_factory",
        "test_autodiff",
        "nondiff_argnums",
        "test_name",
    ],
)


def op_record(
    name, nargs, dtypes, rng_factory, test_grad, nondiff_argnums=(), test_name=None
):
    test_name = test_name or name
    nondiff_argnums = tuple(sorted(set(nondiff_argnums)))
    return OpRecord(
        name, nargs, dtypes, rng_factory, test_grad, nondiff_argnums, test_name
    )


float_dtypes = jtu.dtypes.floating
int_dtypes = jtu.dtypes.integer

# TODO(phawkins): we should probably separate out the function domains used for
# autodiff tests from the function domains used for equivalence testing. For
# example, logit should closely match its scipy equivalent everywhere, but we
# don't expect numerical gradient tests to pass for inputs very close to 0.

JAX_SPECIAL_FUNCTION_RECORDS = [
    op_record("gammaln", 1, float_dtypes, jtu.rand_positive, False),
]


class LaxScipySpcialFunctionsTest(jtu.JaxTestCase):
    def _GetArgsMaker(self, rng, shapes, dtypes):
        return lambda: [rng(shape, dtype) for shape, dtype in zip(shapes, dtypes)]

    @parameterized.parameters(
        itertools.chain.from_iterable(
            jtu.sample_product_testcases(
                [
                    dict(
                        op=rec.name,
                        rng_factory=rec.rng_factory,
                        test_autodiff=rec.test_autodiff,
                        nondiff_argnums=rec.nondiff_argnums,
                    )
                ],
                shapes=itertools.combinations_with_replacement(all_shapes, rec.nargs),
                dtypes=(
                    itertools.combinations_with_replacement(rec.dtypes, rec.nargs)
                    if isinstance(rec.dtypes, list)
                    else itertools.product(*rec.dtypes)
                ),
            )
            for rec in JAX_SPECIAL_FUNCTION_RECORDS
        )
    )
    @jax.numpy_rank_promotion(
        "allow"
    )  # This test explicitly exercises implicit rank promotion.
    @jax.numpy_dtype_promotion(
        "standard"
    )  # This test explicitly exercises dtype promotion
    def testScipySpecialFun(
        self, op, rng_factory, shapes, dtypes, test_autodiff, nondiff_argnums
    ):
        scipy_op = getattr(osp_special, op)
        lax_op = partial_timed(getattr(lsp_special, op))
        rng = rng_factory(self.rng())
        args_maker = self._GetArgsMaker(rng, shapes, dtypes)
        args = args_maker()
        self.assertAllClose(
            scipy_op(*args), lax_op(*args), atol=1e-3, rtol=1e-3, check_dtypes=False
        )
        self._CompileAndCheck(lax_op, args_maker, rtol=1e-4)

        if test_autodiff:

            def partial_lax_op(*vals):
                list_args = list(vals)
                for i in nondiff_argnums:
                    list_args.insert(i, args[i])
                return lax_op(*list_args)

            assert list(nondiff_argnums) == sorted(set(nondiff_argnums))
            diff_args = [x for i, x in enumerate(args) if i not in nondiff_argnums]
            jtu.check_grads(
                partial_lax_op,
                diff_args,
                order=1,
                atol=jtu.if_device_under_test("tpu", 0.1, 1e-3),
                rtol=0.1,
                eps=1e-3,
            )


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
