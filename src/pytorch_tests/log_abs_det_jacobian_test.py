# Owner(s): ["module: distributions"]

"""
Note [Randomized statistical tests]
-----------------------------------

This note describes how to maintain tests in this file as random sources
change. This file contains two types of randomized tests:

1. The easier type of randomized test are tests that should always pass but are
   initialized with random data. If these fail something is wrong, but it's
   fine to use a fixed seed by inheriting from common.TestCase.

2. The trickier tests are statistical tests. These tests explicitly call
   set_rng_seed(n) and are marked "see Note [Randomized statistical tests]".
   These statistical tests have a known positive failure rate
   (we set failure_rate=1e-3 by default). We need to balance strength of these
   tests with annoyance of false alarms. One way that works is to specifically
   set seeds in each of the randomized tests. When a random generator
   occasionally changes (as in #4312 vectorizing the Box-Muller sampler), some
   of these statistical tests may (rarely) fail. If one fails in this case,
   it's fine to increment the seed of the failing test (but you shouldn't need
   to increment it more than once; otherwise something is probably actually
   wrong).

3. `test_geometric_sample`, `test_binomial_sample` and `test_poisson_sample`
   are validated against `scipy.stats.` which are not guaranteed to be identical
   across different versions of scipy (namely, they yield invalid results in 1.7+)
"""

import math
import numbers
import unittest
from collections import namedtuple
from itertools import product
from random import shuffle
from packaging import version

import torch

# TODO: remove this global setting
# Distributions tests use double as the default dtype
torch.set_default_dtype(torch.double)

from torch._six import inf, nan
from torch.testing._internal.common_utils import \
    (TestCase, run_tests, set_rng_seed, TEST_WITH_UBSAN, load_tests,
     gradcheck)
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.autograd import grad
import torch.autograd.forward_ad as fwAD
from torch.autograd.functional import jacobian
from torch.distributions import (Bernoulli, Beta, Binomial, Categorical,
                                 Cauchy, Chi2, ContinuousBernoulli, Dirichlet,
                                 Distribution, Exponential, ExponentialFamily,
                                 FisherSnedecor, Gamma, Geometric, Gumbel,
                                 HalfCauchy, HalfNormal, Independent, Kumaraswamy,
                                 LKJCholesky, Laplace, LogisticNormal,
                                 LogNormal, LowRankMultivariateNormal,
                                 MixtureSameFamily, Multinomial, MultivariateNormal,
                                 NegativeBinomial, Normal,
                                 OneHotCategorical, OneHotCategoricalStraightThrough,
                                 Pareto, Poisson, RelaxedBernoulli, RelaxedOneHotCategorical,
                                 StudentT, TransformedDistribution, Uniform,
                                 VonMises, Weibull, Wishart, constraints, kl_divergence)
from torch.distributions.constraint_registry import transform_to
from torch.distributions.constraints import Constraint, is_dependent
from torch.distributions.dirichlet import _Dirichlet_backward
from torch.distributions.kl import _kl_expfamily_expfamily
from torch.distributions.transforms import (AffineTransform, CatTransform, ExpTransform,
                                            StackTransform, identity_transform)
from torch.distributions.utils import (probs_to_logits, lazy_property, tril_matrix_to_vec,
                                       vec_to_tril_matrix)
from torch.nn.functional import softmax

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests



class DistributionsTestCase(TestCase):
    def setUp(self):
        """The tests assume that the validation flag is set."""
        torch.distributions.Distribution.set_default_validate_args(True)
        super(DistributionsTestCase, self).setUp()


class TestFunctors(DistributionsTestCase):
    def test_cat_transform(self):
        x1 = -1 * torch.arange(1, 101, dtype=torch.float).view(-1, 100)
        x2 = (torch.arange(1, 101, dtype=torch.float).view(-1, 100) - 1) / 100
        x3 = torch.arange(1, 101, dtype=torch.float).view(-1, 100)
        t1, t2, t3 = ExpTransform(), AffineTransform(1, 100), identity_transform
        dim = 0
        x = torch.cat([x1, x2, x3], dim=dim)
        t = CatTransform([t1, t2, t3], dim=dim)
        actual_dom_check = t.domain.check(x)
        expected_dom_check = torch.cat([t1.domain.check(x1),
                                        t2.domain.check(x2),
                                        t3.domain.check(x3)], dim=dim)
        self.assertEqual(expected_dom_check, actual_dom_check)
        actual = t(x)
        expected = torch.cat([t1(x1), t2(x2), t3(x3)], dim=dim)
        self.assertEqual(expected, actual)
        y1 = torch.arange(1, 101, dtype=torch.float).view(-1, 100)
        y2 = torch.arange(1, 101, dtype=torch.float).view(-1, 100)
        y3 = torch.arange(1, 101, dtype=torch.float).view(-1, 100)
        y = torch.cat([y1, y2, y3], dim=dim)
        actual_cod_check = t.codomain.check(y)
        expected_cod_check = torch.cat([t1.codomain.check(y1),
                                        t2.codomain.check(y2),
                                        t3.codomain.check(y3)], dim=dim)
        self.assertEqual(actual_cod_check, expected_cod_check)
        actual_inv = t.inv(y)
        expected_inv = torch.cat([t1.inv(y1), t2.inv(y2), t3.inv(y3)], dim=dim)
        self.assertEqual(expected_inv, actual_inv)
        actual_jac = t.log_abs_det_jacobian(x, y)
        expected_jac = torch.cat([t1.log_abs_det_jacobian(x1, y1),
                                  t2.log_abs_det_jacobian(x2, y2),
                                  t3.log_abs_det_jacobian(x3, y3)], dim=dim)
        self.assertEqual(actual_jac, expected_jac)

    def test_cat_transform_non_uniform(self):
        x1 = -1 * torch.arange(1, 101, dtype=torch.float).view(-1, 100)
        x2 = torch.cat([(torch.arange(1, 101, dtype=torch.float).view(-1, 100) - 1) / 100,
                        torch.arange(1, 101, dtype=torch.float).view(-1, 100)])
        t1 = ExpTransform()
        t2 = CatTransform([AffineTransform(1, 100), identity_transform], dim=0)
        dim = 0
        x = torch.cat([x1, x2], dim=dim)
        t = CatTransform([t1, t2], dim=dim, lengths=[1, 2])
        actual_dom_check = t.domain.check(x)
        expected_dom_check = torch.cat([t1.domain.check(x1),
                                        t2.domain.check(x2)], dim=dim)
        self.assertEqual(expected_dom_check, actual_dom_check)
        actual = t(x)
        expected = torch.cat([t1(x1), t2(x2)], dim=dim)
        self.assertEqual(expected, actual)
        y1 = torch.arange(1, 101, dtype=torch.float).view(-1, 100)
        y2 = torch.cat([torch.arange(1, 101, dtype=torch.float).view(-1, 100),
                        torch.arange(1, 101, dtype=torch.float).view(-1, 100)])
        y = torch.cat([y1, y2], dim=dim)
        actual_cod_check = t.codomain.check(y)
        expected_cod_check = torch.cat([t1.codomain.check(y1),
                                        t2.codomain.check(y2)], dim=dim)
        self.assertEqual(actual_cod_check, expected_cod_check)
        actual_inv = t.inv(y)
        expected_inv = torch.cat([t1.inv(y1), t2.inv(y2)], dim=dim)
        self.assertEqual(expected_inv, actual_inv)
        actual_jac = t.log_abs_det_jacobian(x, y)
        expected_jac = torch.cat([t1.log_abs_det_jacobian(x1, y1),
                                  t2.log_abs_det_jacobian(x2, y2)], dim=dim)
        self.assertEqual(actual_jac, expected_jac)

    def test_cat_event_dim(self):
        t1 = AffineTransform(0, 2 * torch.ones(2), event_dim=1)
        t2 = AffineTransform(0, 2 * torch.ones(2), event_dim=1)
        dim = 1
        bs = 16
        x1 = torch.randn(bs, 2)
        x2 = torch.randn(bs, 2)
        x = torch.cat([x1, x2], dim=1)
        t = CatTransform([t1, t2], dim=dim, lengths=[2, 2])
        y1 = t1(x1)
        y2 = t2(x2)
        y = t(x)
        actual_jac = t.log_abs_det_jacobian(x, y)
        expected_jac = sum([t1.log_abs_det_jacobian(x1, y1),
                            t2.log_abs_det_jacobian(x2, y2)])

    def test_stack_transform(self):
        x1 = -1 * torch.arange(1, 101, dtype=torch.float)
        x2 = (torch.arange(1, 101, dtype=torch.float) - 1) / 100
        x3 = torch.arange(1, 101, dtype=torch.float)
        t1, t2, t3 = ExpTransform(), AffineTransform(1, 100), identity_transform
        dim = 0
        x = torch.stack([x1, x2, x3], dim=dim)
        t = StackTransform([t1, t2, t3], dim=dim)
        actual_dom_check = t.domain.check(x)
        expected_dom_check = torch.stack([t1.domain.check(x1),
                                          t2.domain.check(x2),
                                          t3.domain.check(x3)], dim=dim)
        self.assertEqual(expected_dom_check, actual_dom_check)
        actual = t(x)
        expected = torch.stack([t1(x1), t2(x2), t3(x3)], dim=dim)
        self.assertEqual(expected, actual)
        y1 = torch.arange(1, 101, dtype=torch.float)
        y2 = torch.arange(1, 101, dtype=torch.float)
        y3 = torch.arange(1, 101, dtype=torch.float)
        y = torch.stack([y1, y2, y3], dim=dim)
        actual_cod_check = t.codomain.check(y)
        expected_cod_check = torch.stack([t1.codomain.check(y1),
                                          t2.codomain.check(y2),
                                          t3.codomain.check(y3)], dim=dim)
        self.assertEqual(actual_cod_check, expected_cod_check)
        actual_inv = t.inv(x)
        expected_inv = torch.stack([t1.inv(x1), t2.inv(x2), t3.inv(x3)], dim=dim)
        self.assertEqual(expected_inv, actual_inv)
        actual_jac = t.log_abs_det_jacobian(x, y)
        expected_jac = torch.stack([t1.log_abs_det_jacobian(x1, y1),
                                    t2.log_abs_det_jacobian(x2, y2),
                                    t3.log_abs_det_jacobian(x3, y3)], dim=dim)
        self.assertEqual(actual_jac, expected_jac)


if __name__ == '__main__' and torch._C.has_lapack:
    run_tests()
