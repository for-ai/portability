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


import itertools
from functools import partial

import jax
import numpy as np
import scipy.stats as osp_stats
import scipy.version
from absl.testing import absltest
from jax._src import dtypes
from jax._src import test_util as jtu
from jax._src import tree_util
from jax.config import config
from jax.scipy import stats as lsp_stats
from jax.scipy.special import expit

from ..utils.timer_wrapper import jax_op_timer, partial_timed

config.parse_flags_with_absl()

scipy_version = tuple(map(int, scipy.version.version.split(".")[:3]))
numpy_version = jtu.numpy_version()

all_shapes = [(), (4,), (3, 4), (3, 1), (1, 4), (2, 1, 4)]
one_and_two_dim_shapes = [(4,), (3, 4), (3, 1), (1, 4)]


def genNamedParametersNArgs(n):
    return jtu.sample_product(
        shapes=itertools.combinations_with_replacement(all_shapes, n),
        dtypes=itertools.combinations_with_replacement(jtu.dtypes.floating, n),
    )


# Allow implicit rank promotion in these tests, as virtually every test exercises it.
@jtu.with_config(jax_numpy_rank_promotion="allow")
class LaxBackedScipyStatsTests(jtu.JaxTestCase):
    """Tests for LAX-backed scipy.stats implementations"""

    @genNamedParametersNArgs(3)
    def testPoissonCdf(self, shapes, dtypes):
        rng = jtu.rand_default(self.rng())
        scipy_fun = osp_stats.poisson.cdf
        timer = jax_op_timer()
        with timer:
            lax_fun = lsp_stats.poisson.cdf
            timer.gen.send(lax_fun)

        def args_maker():
            k, mu, loc = map(rng, shapes, dtypes)
            # clipping to ensure that rate parameter is strictly positive
            mu = np.clip(np.abs(mu), a_min=0.1, a_max=None).astype(mu.dtype)
            return [k, mu, loc]

        with jtu.strict_promotion_if_dtypes_match(dtypes):
            self._CheckAgainstNumpy(
                scipy_fun, lax_fun, args_maker, check_dtypes=False, tol=1e-3
            )
            self._CompileAndCheck(lax_fun, args_maker)

    @genNamedParametersNArgs(2)
    def testBernoulliCdf(self, shapes, dtypes):
        rng_int = jtu.rand_int(self.rng(), -100, 100)
        rng_uniform = jtu.rand_uniform(self.rng())
        scipy_fun = osp_stats.bernoulli.cdf
        timer = jax_op_timer()
        with timer:
            lax_fun = lsp_stats.bernoulli.cdf
            timer.gen.send(lax_fun)

        def args_maker():
            x = rng_int(shapes[0], dtypes[0])
            p = rng_uniform(shapes[1], dtypes[1])
            return [x, p]

        with jtu.strict_promotion_if_dtypes_match(dtypes):
            self._CheckAgainstNumpy(
                scipy_fun, lax_fun, args_maker, check_dtypes=False, tol=5e-4
            )
            self._CompileAndCheck(lax_fun, args_maker)

    @genNamedParametersNArgs(5)
    def testBetaLogCdf(self, shapes, dtypes):
        rng = jtu.rand_positive(self.rng())
        scipy_fun = osp_stats.beta.logcdf
        lax_fun = lsp_stats.beta.logcdf

        def args_maker():
            x, a, b, loc, scale = map(rng, shapes, dtypes)
            return [x, a, b, loc, scale]

        with jtu.strict_promotion_if_dtypes_match(dtypes):
            self._CheckAgainstNumpy(
                scipy_fun, lax_fun, args_maker, check_dtypes=False, tol=1e-3
            )
            self._CompileAndCheck(
                lax_fun, args_maker, rtol={np.float32: 2e-3, np.float64: 1e-4}
            )

    @genNamedParametersNArgs(3)
    def testCauchyLogCdf(self, shapes, dtypes):
        rng = jtu.rand_default(self.rng())
        scipy_fun = osp_stats.cauchy.logcdf
        lax_fun = lsp_stats.cauchy.logcdf

        def args_maker():
            x, loc, scale = map(rng, shapes, dtypes)
            # clipping to ensure that scale is not too low
            scale = np.clip(np.abs(scale), a_min=0.1, a_max=None).astype(scale.dtype)
            return [x, loc, scale]

        with jtu.strict_promotion_if_dtypes_match(dtypes):
            self._CheckAgainstNumpy(
                scipy_fun, lax_fun, args_maker, check_dtypes=False, tol=1e-4
            )
            self._CompileAndCheck(lax_fun, args_maker)

    @genNamedParametersNArgs(4)
    def testGammaLogCdf(self, shapes, dtypes):
        rng = jtu.rand_positive(self.rng())
        scipy_fun = osp_stats.gamma.logcdf
        lax_fun = lsp_stats.gamma.logcdf

        def args_maker():
            x, a, loc, scale = map(rng, shapes, dtypes)
            x = np.clip(x, 0, None).astype(x.dtype)
            return [x, a, loc, scale]

        with jtu.strict_promotion_if_dtypes_match(dtypes):
            self._CheckAgainstNumpy(
                scipy_fun, lax_fun, args_maker, check_dtypes=False, tol=5e-4
            )
            self._CompileAndCheck(lax_fun, args_maker)

    @genNamedParametersNArgs(2)
    def testGenNormCdf(self, shapes, dtypes):
        rng = jtu.rand_default(self.rng())
        scipy_fun = osp_stats.gennorm.cdf
        timer = jax_op_timer()
        with timer:
            lax_fun = lsp_stats.gennorm.cdf
            timer.gen.send(lax_fun)

        def args_maker():
            x, p = map(rng, shapes, dtypes)
            return [x, p]

        with jtu.strict_promotion_if_dtypes_match(dtypes):
            self._CheckAgainstNumpy(
                scipy_fun, lax_fun, args_maker, check_dtypes=False, tol=1e-4, rtol=1e-3
            )
            self._CompileAndCheck(lax_fun, args_maker)

    @genNamedParametersNArgs(3)
    def testLaplaceCdf(self, shapes, dtypes):
        rng = jtu.rand_default(self.rng())
        scipy_fun = osp_stats.laplace.cdf
        timer = jax_op_timer()
        with timer:
            lax_fun = lsp_stats.laplace.cdf
            timer.gen.send(lax_fun)

        def args_maker():
            x, loc, scale = map(rng, shapes, dtypes)
            # ensure that scale is not too low
            scale = np.clip(scale, a_min=0.1, a_max=None).astype(scale.dtype)
            return [x, loc, scale]

        with jtu.strict_promotion_if_dtypes_match(dtypes):
            self._CheckAgainstNumpy(
                scipy_fun,
                lax_fun,
                args_maker,
                check_dtypes=False,
                tol={np.float32: 1e-5, np.float64: 1e-6},
            )
            self._CompileAndCheck(lax_fun, args_maker)

    @genNamedParametersNArgs(3)
    def testLogisticCdf(self, shapes, dtypes):
        rng = jtu.rand_default(self.rng())
        scipy_fun = osp_stats.logistic.cdf
        timer = jax_op_timer()
        with timer:
            lax_fun = lsp_stats.logistic.cdf
            timer.gen.send(lax_fun)

        def args_maker():
            x, loc, scale = map(rng, shapes, dtypes)
            # ensure that scale is not too low
            scale = np.clip(scale, a_min=0.1, a_max=None).astype(scale.dtype)
            return [x, loc, scale]

        with jtu.strict_promotion_if_dtypes_match(dtypes):
            self._CheckAgainstNumpy(
                scipy_fun, lax_fun, args_maker, check_dtypes=False, tol=3e-5
            )
            self._CompileAndCheck(lax_fun, args_maker)

    @genNamedParametersNArgs(3)
    def testNormLogCdf(self, shapes, dtypes):
        rng = jtu.rand_default(self.rng())
        scipy_fun = osp_stats.norm.logcdf
        lax_fun = lsp_stats.norm.logcdf

        def args_maker():
            x, loc, scale = map(rng, shapes, dtypes)
            # clipping to ensure that scale is not too low
            scale = np.clip(np.abs(scale), a_min=0.1, a_max=None).astype(scale.dtype)
            return [x, loc, scale]

        with jtu.strict_promotion_if_dtypes_match(dtypes):
            self._CheckAgainstNumpy(
                scipy_fun, lax_fun, args_maker, check_dtypes=False, tol=1e-4
            )
            self._CompileAndCheck(lax_fun, args_maker)

    @genNamedParametersNArgs(3)
    def testNormCdf(self, shapes, dtypes):
        rng = jtu.rand_default(self.rng())
        scipy_fun = osp_stats.norm.cdf
        timer = jax_op_timer()
        with timer:
            lax_fun = lsp_stats.norm.cdf
            timer.gen.send(lax_fun)

        def args_maker():
            x, loc, scale = map(rng, shapes, dtypes)
            # clipping to ensure that scale is not too low
            scale = np.clip(np.abs(scale), a_min=0.1, a_max=None).astype(scale.dtype)
            return [x, loc, scale]

        with jtu.strict_promotion_if_dtypes_match(dtypes):
            self._CheckAgainstNumpy(
                scipy_fun, lax_fun, args_maker, check_dtypes=False, tol=1e-6
            )
            self._CompileAndCheck(lax_fun, args_maker)

    @genNamedParametersNArgs(5)
    def testTruncnormLogCdf(self, shapes, dtypes):
        rng = jtu.rand_default(self.rng())
        scipy_fun = osp_stats.truncnorm.logcdf
        lax_fun = lsp_stats.truncnorm.logcdf

        def args_maker():
            x, a, b, loc, scale = map(rng, shapes, dtypes)

            # clipping to ensure that scale is not too low
            scale = np.clip(np.abs(scale), a_min=0.1, a_max=None).astype(scale.dtype)
            return [x, a, b, loc, scale]

        with jtu.strict_promotion_if_dtypes_match(dtypes):
            self._CheckAgainstNumpy(
                scipy_fun, lax_fun, args_maker, check_dtypes=False, tol=1e-3
            )
            self._CompileAndCheck(lax_fun, args_maker)

    @genNamedParametersNArgs(5)
    def testTruncnormCdf(self, shapes, dtypes):
        rng = jtu.rand_default(self.rng())
        scipy_fun = osp_stats.truncnorm.cdf
        timer = jax_op_timer()
        with timer:
            lax_fun = lsp_stats.truncnorm.cdf
            timer.gen.send(lax_fun)

        def args_maker():
            x, a, b, loc, scale = map(rng, shapes, dtypes)

            # clipping to ensure that scale is not too low
            scale = np.clip(np.abs(scale), a_min=0.1, a_max=None).astype(scale.dtype)
            return [x, a, b, loc, scale]

        with jtu.strict_promotion_if_dtypes_match(dtypes):
            self._CheckAgainstNumpy(
                scipy_fun, lax_fun, args_maker, check_dtypes=False, tol=1e-3
            )
            self._CompileAndCheck(lax_fun, args_maker)

    @genNamedParametersNArgs(4)
    def testChi2LogCdf(self, shapes, dtypes):
        rng = jtu.rand_positive(self.rng())
        scipy_fun = osp_stats.chi2.logcdf
        lax_fun = lsp_stats.chi2.logcdf

        def args_maker():
            x, df, loc, scale = map(rng, shapes, dtypes)
            return [x, df, loc, scale]

        with jtu.strict_promotion_if_dtypes_match(dtypes):
            self._CheckAgainstNumpy(
                scipy_fun, lax_fun, args_maker, check_dtypes=False, tol=5e-4
            )
            self._CompileAndCheck(lax_fun, args_maker)
