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

TEST_NUMPY = True
try:
    import numpy as np
    import scipy.stats
    import scipy.special
except ImportError:
    TEST_NUMPY = False


def pairwise(Dist, *params):
    """
    Creates a pair of distributions `Dist` initialized to test each element of
    param with each other.
    """
    params1 = [torch.tensor([p] * len(p)) for p in params]
    params2 = [p.transpose(0, 1) for p in params1]
    return Dist(*params1), Dist(*params2)


def is_all_nan(tensor):
    """
    Checks if all entries of a tensor is nan.
    """
    return (tensor != tensor).all()


# Register all distributions for generic tests.
Example = namedtuple('Example', ['Dist', 'params'])
EXAMPLES = [
    Example(Bernoulli, [
        {'probs': torch.tensor([0.7, 0.2, 0.4], requires_grad=True)},
        {'probs': torch.tensor([0.3], requires_grad=True)},
        {'probs': 0.3},
        {'logits': torch.tensor([0.], requires_grad=True)},
    ]),
    Example(Geometric, [
        {'probs': torch.tensor([0.7, 0.2, 0.4], requires_grad=True)},
        {'probs': torch.tensor([0.3], requires_grad=True)},
        {'probs': 0.3},
    ]),
    Example(Beta, [
        {
            'concentration1': torch.randn(2, 3).exp().requires_grad_(),
            'concentration0': torch.randn(2, 3).exp().requires_grad_(),
        },
        {
            'concentration1': torch.randn(4).exp().requires_grad_(),
            'concentration0': torch.randn(4).exp().requires_grad_(),
        },
    ]),
    Example(Categorical, [
        {'probs': torch.tensor([[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True)},
        {'probs': torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)},
        {'logits': torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)},
    ]),
    Example(Binomial, [
        {'probs': torch.tensor([[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True), 'total_count': 10},
        {'probs': torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True), 'total_count': 10},
        {'probs': torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True), 'total_count': torch.tensor([10])},
        {'probs': torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True), 'total_count': torch.tensor([10, 8])},
        {'probs': torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True),
         'total_count': torch.tensor([[10., 8.], [5., 3.]])},
        {'probs': torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True),
         'total_count': torch.tensor(0.)},
    ]),
    Example(NegativeBinomial, [
        {'probs': torch.tensor([[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True), 'total_count': 10},
        {'probs': torch.tensor([[0.9, 0.0], [0.0, 0.9]], requires_grad=True), 'total_count': 10},
        {'probs': torch.tensor([[0.9, 0.0], [0.0, 0.9]], requires_grad=True), 'total_count': torch.tensor([10])},
        {'probs': torch.tensor([[0.9, 0.0], [0.0, 0.9]], requires_grad=True), 'total_count': torch.tensor([10, 8])},
        {'probs': torch.tensor([[0.9, 0.0], [0.0, 0.9]], requires_grad=True),
         'total_count': torch.tensor([[10., 8.], [5., 3.]])},
        {'probs': torch.tensor([[0.9, 0.0], [0.0, 0.9]], requires_grad=True),
         'total_count': torch.tensor(0.)},
    ]),
    Example(Multinomial, [
        {'probs': torch.tensor([[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True), 'total_count': 10},
        {'probs': torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True), 'total_count': 10},
    ]),
    Example(Cauchy, [
        {'loc': 0.0, 'scale': 1.0},
        {'loc': torch.tensor([0.0]), 'scale': 1.0},
        {'loc': torch.tensor([[0.0], [0.0]]),
         'scale': torch.tensor([[1.0], [1.0]])}
    ]),
    Example(Chi2, [
        {'df': torch.randn(2, 3).exp().requires_grad_()},
        {'df': torch.randn(1).exp().requires_grad_()},
    ]),
    Example(StudentT, [
        {'df': torch.randn(2, 3).exp().requires_grad_()},
        {'df': torch.randn(1).exp().requires_grad_()},
    ]),
    Example(Dirichlet, [
        {'concentration': torch.randn(2, 3).exp().requires_grad_()},
        {'concentration': torch.randn(4).exp().requires_grad_()},
    ]),
    Example(Exponential, [
        {'rate': torch.randn(5, 5).abs().requires_grad_()},
        {'rate': torch.randn(1).abs().requires_grad_()},
    ]),
    Example(FisherSnedecor, [
        {
            'df1': torch.randn(5, 5).abs().requires_grad_(),
            'df2': torch.randn(5, 5).abs().requires_grad_(),
        },
        {
            'df1': torch.randn(1).abs().requires_grad_(),
            'df2': torch.randn(1).abs().requires_grad_(),
        },
        {
            'df1': torch.tensor([1.0]),
            'df2': 1.0,
        }
    ]),
    Example(Gamma, [
        {
            'concentration': torch.randn(2, 3).exp().requires_grad_(),
            'rate': torch.randn(2, 3).exp().requires_grad_(),
        },
        {
            'concentration': torch.randn(1).exp().requires_grad_(),
            'rate': torch.randn(1).exp().requires_grad_(),
        },
    ]),
    Example(Gumbel, [
        {
            'loc': torch.randn(5, 5, requires_grad=True),
            'scale': torch.randn(5, 5).abs().requires_grad_(),
        },
        {
            'loc': torch.randn(1, requires_grad=True),
            'scale': torch.randn(1).abs().requires_grad_(),
        },
    ]),
    Example(HalfCauchy, [
        {'scale': 1.0},
        {'scale': torch.tensor([[1.0], [1.0]])}
    ]),
    Example(HalfNormal, [
        {'scale': torch.randn(5, 5).abs().requires_grad_()},
        {'scale': torch.randn(1).abs().requires_grad_()},
        {'scale': torch.tensor([1e-5, 1e-5], requires_grad=True)}
    ]),
    Example(Independent, [
        {
            'base_distribution': Normal(torch.randn(2, 3, requires_grad=True),
                                        torch.randn(2, 3).abs().requires_grad_()),
            'reinterpreted_batch_ndims': 0,
        },
        {
            'base_distribution': Normal(torch.randn(2, 3, requires_grad=True),
                                        torch.randn(2, 3).abs().requires_grad_()),
            'reinterpreted_batch_ndims': 1,
        },
        {
            'base_distribution': Normal(torch.randn(2, 3, requires_grad=True),
                                        torch.randn(2, 3).abs().requires_grad_()),
            'reinterpreted_batch_ndims': 2,
        },
        {
            'base_distribution': Normal(torch.randn(2, 3, 5, requires_grad=True),
                                        torch.randn(2, 3, 5).abs().requires_grad_()),
            'reinterpreted_batch_ndims': 2,
        },
        {
            'base_distribution': Normal(torch.randn(2, 3, 5, requires_grad=True),
                                        torch.randn(2, 3, 5).abs().requires_grad_()),
            'reinterpreted_batch_ndims': 3,
        },
    ]),
    Example(Kumaraswamy, [
        {
            'concentration1': torch.empty(2, 3).uniform_(1, 2).requires_grad_(),
            'concentration0': torch.empty(2, 3).uniform_(1, 2).requires_grad_(),
        },
        {
            'concentration1': torch.rand(4).uniform_(1, 2).requires_grad_(),
            'concentration0': torch.rand(4).uniform_(1, 2).requires_grad_(),
        },
    ]),
    Example(LKJCholesky, [
        {
            'dim': 2,
            'concentration': 0.5
        },
        {
            'dim': 3,
            'concentration': torch.tensor([0.5, 1., 2.]),
        },
        {
            'dim': 100,
            'concentration': 4.
        },
    ]),
    Example(Laplace, [
        {
            'loc': torch.randn(5, 5, requires_grad=True),
            'scale': torch.randn(5, 5).abs().requires_grad_(),
        },
        {
            'loc': torch.randn(1, requires_grad=True),
            'scale': torch.randn(1).abs().requires_grad_(),
        },
        {
            'loc': torch.tensor([1.0, 0.0], requires_grad=True),
            'scale': torch.tensor([1e-5, 1e-5], requires_grad=True),
        },
    ]),
    Example(LogNormal, [
        {
            'loc': torch.randn(5, 5, requires_grad=True),
            'scale': torch.randn(5, 5).abs().requires_grad_(),
        },
        {
            'loc': torch.randn(1, requires_grad=True),
            'scale': torch.randn(1).abs().requires_grad_(),
        },
        {
            'loc': torch.tensor([1.0, 0.0], requires_grad=True),
            'scale': torch.tensor([1e-5, 1e-5], requires_grad=True),
        },
    ]),
    Example(LogisticNormal, [
        {
            'loc': torch.randn(5, 5).requires_grad_(),
            'scale': torch.randn(5, 5).abs().requires_grad_(),
        },
        {
            'loc': torch.randn(1).requires_grad_(),
            'scale': torch.randn(1).abs().requires_grad_(),
        },
        {
            'loc': torch.tensor([1.0, 0.0], requires_grad=True),
            'scale': torch.tensor([1e-5, 1e-5], requires_grad=True),
        },
    ]),
    Example(LowRankMultivariateNormal, [
        {
            'loc': torch.randn(5, 2, requires_grad=True),
            'cov_factor': torch.randn(5, 2, 1, requires_grad=True),
            'cov_diag': torch.tensor([2.0, 0.25], requires_grad=True),
        },
        {
            'loc': torch.randn(4, 3, requires_grad=True),
            'cov_factor': torch.randn(3, 2, requires_grad=True),
            'cov_diag': torch.tensor([5.0, 1.5, 3.], requires_grad=True),
        }
    ]),
    Example(MultivariateNormal, [
        {
            'loc': torch.randn(5, 2, requires_grad=True),
            'covariance_matrix': torch.tensor([[2.0, 0.3], [0.3, 0.25]], requires_grad=True),
        },
        {
            'loc': torch.randn(2, 3, requires_grad=True),
            'precision_matrix': torch.tensor([[2.0, 0.1, 0.0],
                                              [0.1, 0.25, 0.0],
                                              [0.0, 0.0, 0.3]], requires_grad=True),
        },
        {
            'loc': torch.randn(5, 3, 2, requires_grad=True),
            'scale_tril': torch.tensor([[[2.0, 0.0], [-0.5, 0.25]],
                                        [[2.0, 0.0], [0.3, 0.25]],
                                        [[5.0, 0.0], [-0.5, 1.5]]], requires_grad=True),
        },
        {
            'loc': torch.tensor([1.0, -1.0]),
            'covariance_matrix': torch.tensor([[5.0, -0.5], [-0.5, 1.5]]),
        },
    ]),
    Example(Normal, [
        {
            'loc': torch.randn(5, 5, requires_grad=True),
            'scale': torch.randn(5, 5).abs().requires_grad_(),
        },
        {
            'loc': torch.randn(1, requires_grad=True),
            'scale': torch.randn(1).abs().requires_grad_(),
        },
        {
            'loc': torch.tensor([1.0, 0.0], requires_grad=True),
            'scale': torch.tensor([1e-5, 1e-5], requires_grad=True),
        },
    ]),
    Example(OneHotCategorical, [
        {'probs': torch.tensor([[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True)},
        {'probs': torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)},
        {'logits': torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)},
    ]),
    Example(OneHotCategoricalStraightThrough, [
        {'probs': torch.tensor([[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True)},
        {'probs': torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)},
        {'logits': torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)},
    ]),
    Example(Pareto, [
        {
            'scale': 1.0,
            'alpha': 1.0
        },
        {
            'scale': torch.randn(5, 5).abs().requires_grad_(),
            'alpha': torch.randn(5, 5).abs().requires_grad_()
        },
        {
            'scale': torch.tensor([1.0]),
            'alpha': 1.0
        }
    ]),
    Example(Poisson, [
        {
            'rate': torch.randn(5, 5).abs().requires_grad_(),
        },
        {
            'rate': torch.randn(3).abs().requires_grad_(),
        },
        {
            'rate': 0.2,
        },
        {
            'rate': torch.tensor([0.0], requires_grad=True),
        },
        {
            'rate': 0.0,
        }
    ]),
    Example(RelaxedBernoulli, [
        {
            'temperature': torch.tensor([0.5], requires_grad=True),
            'probs': torch.tensor([0.7, 0.2, 0.4], requires_grad=True),
        },
        {
            'temperature': torch.tensor([2.0]),
            'probs': torch.tensor([0.3]),
        },
        {
            'temperature': torch.tensor([7.2]),
            'logits': torch.tensor([-2.0, 2.0, 1.0, 5.0])
        }
    ]),
    Example(RelaxedOneHotCategorical, [
        {
            'temperature': torch.tensor([0.5], requires_grad=True),
            'probs': torch.tensor([[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]], requires_grad=True)
        },
        {
            'temperature': torch.tensor([2.0]),
            'probs': torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        },
        {
            'temperature': torch.tensor([7.2]),
            'logits': torch.tensor([[-2.0, 2.0], [1.0, 5.0]])
        }
    ]),
    Example(TransformedDistribution, [
        {
            'base_distribution': Normal(torch.randn(2, 3, requires_grad=True),
                                        torch.randn(2, 3).abs().requires_grad_()),
            'transforms': [],
        },
        {
            'base_distribution': Normal(torch.randn(2, 3, requires_grad=True),
                                        torch.randn(2, 3).abs().requires_grad_()),
            'transforms': ExpTransform(),
        },
        {
            'base_distribution': Normal(torch.randn(2, 3, 5, requires_grad=True),
                                        torch.randn(2, 3, 5).abs().requires_grad_()),
            'transforms': [AffineTransform(torch.randn(3, 5), torch.randn(3, 5)),
                           ExpTransform()],
        },
        {
            'base_distribution': Normal(torch.randn(2, 3, 5, requires_grad=True),
                                        torch.randn(2, 3, 5).abs().requires_grad_()),
            'transforms': AffineTransform(1, 2),
        },
        {
            'base_distribution': Uniform(torch.tensor(1e8).log(), torch.tensor(1e10).log()),
            'transforms': ExpTransform(),
        },
    ]),
    Example(Uniform, [
        {
            'low': torch.zeros(5, 5, requires_grad=True),
            'high': torch.ones(5, 5, requires_grad=True),
        },
        {
            'low': torch.zeros(1, requires_grad=True),
            'high': torch.ones(1, requires_grad=True),
        },
        {
            'low': torch.tensor([1.0, 1.0], requires_grad=True),
            'high': torch.tensor([2.0, 3.0], requires_grad=True),
        },
    ]),
    Example(Weibull, [
        {
            'scale': torch.randn(5, 5).abs().requires_grad_(),
            'concentration': torch.randn(1).abs().requires_grad_()
        }
    ]),
    Example(Wishart, [
        {
            'covariance_matrix': torch.tensor([[2.0, 0.3], [0.3, 0.25]], requires_grad=True),
            'df': torch.tensor([3.], requires_grad=True),
        },
        {
            'precision_matrix': torch.tensor([[2.0, 0.1, 0.0],
                                              [0.1, 0.25, 0.0],
                                              [0.0, 0.0, 0.3]], requires_grad=True),
            'df': torch.tensor([5., 4], requires_grad=True),
        },
        {
            'scale_tril': torch.tensor([[[2.0, 0.0], [-0.5, 0.25]],
                                        [[2.0, 0.0], [0.3, 0.25]],
                                        [[5.0, 0.0], [-0.5, 1.5]]], requires_grad=True),
            'df': torch.tensor([5., 3.5, 3], requires_grad=True),
        },
        {
            'covariance_matrix': torch.tensor([[5.0, -0.5], [-0.5, 1.5]]),
            'df': torch.tensor([3.0]),
        },
        {
            'covariance_matrix': torch.tensor([[5.0, -0.5], [-0.5, 1.5]]),
            'df': 3.0,
        },
    ]),
    Example(MixtureSameFamily, [
        {
            'mixture_distribution': Categorical(torch.rand(5, requires_grad=True)),
            'component_distribution': Normal(torch.randn(5, requires_grad=True),
                                             torch.rand(5, requires_grad=True)),
        },
        {
            'mixture_distribution': Categorical(torch.rand(5, requires_grad=True)),
            'component_distribution': MultivariateNormal(
                loc=torch.randn(5, 2, requires_grad=True),
                covariance_matrix=torch.tensor([[2.0, 0.3], [0.3, 0.25]], requires_grad=True)),
        },
    ]),
    Example(VonMises, [
        {
            'loc': torch.tensor(1.0, requires_grad=True),
            'concentration': torch.tensor(10.0, requires_grad=True)
        },
        {
            'loc': torch.tensor([0.0, math.pi / 2], requires_grad=True),
            'concentration': torch.tensor([1.0, 10.0], requires_grad=True)
        },
    ]),
    Example(ContinuousBernoulli, [
        {'probs': torch.tensor([0.7, 0.2, 0.4], requires_grad=True)},
        {'probs': torch.tensor([0.3], requires_grad=True)},
        {'probs': 0.3},
        {'logits': torch.tensor([0.], requires_grad=True)},
    ])
]

BAD_EXAMPLES = [
    Example(Bernoulli, [
        {'probs': torch.tensor([1.1, 0.2, 0.4], requires_grad=True)},
        {'probs': torch.tensor([-0.5], requires_grad=True)},
        {'probs': 1.00001},
    ]),
    Example(Beta, [
        {
            'concentration1': torch.tensor([0.0], requires_grad=True),
            'concentration0': torch.tensor([0.0], requires_grad=True),
        },
        {
            'concentration1': torch.tensor([-1.0], requires_grad=True),
            'concentration0': torch.tensor([-2.0], requires_grad=True),
        },
    ]),
    Example(Geometric, [
        {'probs': torch.tensor([1.1, 0.2, 0.4], requires_grad=True)},
        {'probs': torch.tensor([-0.3], requires_grad=True)},
        {'probs': 1.00000001},
    ]),
    Example(Categorical, [
        {'probs': torch.tensor([[-0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True)},
        {'probs': torch.tensor([[-1.0, 10.0], [0.0, -1.0]], requires_grad=True)},
    ]),
    Example(Binomial, [
        {'probs': torch.tensor([[-0.0000001, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True),
         'total_count': 10},
        {'probs': torch.tensor([[1.0, 0.0], [0.0, 2.0]], requires_grad=True),
         'total_count': 10},
    ]),
    Example(NegativeBinomial, [
        {'probs': torch.tensor([[-0.0000001, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True),
         'total_count': 10},
        {'probs': torch.tensor([[1.0, 0.0], [0.0, 2.0]], requires_grad=True),
         'total_count': 10},
    ]),
    Example(Cauchy, [
        {'loc': 0.0, 'scale': -1.0},
        {'loc': torch.tensor([0.0]), 'scale': 0.0},
        {'loc': torch.tensor([[0.0], [-2.0]]),
         'scale': torch.tensor([[-0.000001], [1.0]])}
    ]),
    Example(Chi2, [
        {'df': torch.tensor([0.], requires_grad=True)},
        {'df': torch.tensor([-2.], requires_grad=True)},
    ]),
    Example(StudentT, [
        {'df': torch.tensor([0.], requires_grad=True)},
        {'df': torch.tensor([-2.], requires_grad=True)},
    ]),
    Example(Dirichlet, [
        {'concentration': torch.tensor([0.], requires_grad=True)},
        {'concentration': torch.tensor([-2.], requires_grad=True)}
    ]),
    Example(Exponential, [
        {'rate': torch.tensor([0., 0.], requires_grad=True)},
        {'rate': torch.tensor([-2.], requires_grad=True)}
    ]),
    Example(FisherSnedecor, [
        {
            'df1': torch.tensor([0., 0.], requires_grad=True),
            'df2': torch.tensor([-1., -100.], requires_grad=True),
        },
        {
            'df1': torch.tensor([1., 1.], requires_grad=True),
            'df2': torch.tensor([0., 0.], requires_grad=True),
        }
    ]),
    Example(Gamma, [
        {
            'concentration': torch.tensor([0., 0.], requires_grad=True),
            'rate': torch.tensor([-1., -100.], requires_grad=True),
        },
        {
            'concentration': torch.tensor([1., 1.], requires_grad=True),
            'rate': torch.tensor([0., 0.], requires_grad=True),
        }
    ]),
    Example(Gumbel, [
        {
            'loc': torch.tensor([1., 1.], requires_grad=True),
            'scale': torch.tensor([0., 1.], requires_grad=True),
        },
        {
            'loc': torch.tensor([1., 1.], requires_grad=True),
            'scale': torch.tensor([1., -1.], requires_grad=True),
        },
    ]),
    Example(HalfCauchy, [
        {'scale': -1.0},
        {'scale': 0.0},
        {'scale': torch.tensor([[-0.000001], [1.0]])}
    ]),
    Example(HalfNormal, [
        {'scale': torch.tensor([0., 1.], requires_grad=True)},
        {'scale': torch.tensor([1., -1.], requires_grad=True)},
    ]),
    Example(LKJCholesky, [
        {
            'dim': -2,
            'concentration': 0.1
        },
        {
            'dim': 1,
            'concentration': 2.,
        },
        {
            'dim': 2,
            'concentration': 0.,
        },
    ]),
    Example(Laplace, [
        {
            'loc': torch.tensor([1., 1.], requires_grad=True),
            'scale': torch.tensor([0., 1.], requires_grad=True),
        },
        {
            'loc': torch.tensor([1., 1.], requires_grad=True),
            'scale': torch.tensor([1., -1.], requires_grad=True),
        },
    ]),
    Example(LogNormal, [
        {
            'loc': torch.tensor([1., 1.], requires_grad=True),
            'scale': torch.tensor([0., 1.], requires_grad=True),
        },
        {
            'loc': torch.tensor([1., 1.], requires_grad=True),
            'scale': torch.tensor([1., -1.], requires_grad=True),
        },
    ]),
    Example(MultivariateNormal, [
        {
            'loc': torch.tensor([1., 1.], requires_grad=True),
            'covariance_matrix': torch.tensor([[1.0, 0.0], [0.0, -2.0]], requires_grad=True),
        },
    ]),
    Example(Normal, [
        {
            'loc': torch.tensor([1., 1.], requires_grad=True),
            'scale': torch.tensor([0., 1.], requires_grad=True),
        },
        {
            'loc': torch.tensor([1., 1.], requires_grad=True),
            'scale': torch.tensor([1., -1.], requires_grad=True),
        },
        {
            'loc': torch.tensor([1.0, 0.0], requires_grad=True),
            'scale': torch.tensor([1e-5, -1e-5], requires_grad=True),
        },
    ]),
    Example(OneHotCategorical, [
        {'probs': torch.tensor([[0.1, 0.2, 0.3], [0.1, -10.0, 0.2]], requires_grad=True)},
        {'probs': torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)},
    ]),
    Example(OneHotCategoricalStraightThrough, [
        {'probs': torch.tensor([[0.1, 0.2, 0.3], [0.1, -10.0, 0.2]], requires_grad=True)},
        {'probs': torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)},
    ]),
    Example(Pareto, [
        {
            'scale': 0.0,
            'alpha': 0.0
        },
        {
            'scale': torch.tensor([0.0, 0.0], requires_grad=True),
            'alpha': torch.tensor([-1e-5, 0.0], requires_grad=True)
        },
        {
            'scale': torch.tensor([1.0]),
            'alpha': -1.0
        }
    ]),
    Example(Poisson, [
        {
            'rate': torch.tensor([-0.1], requires_grad=True),
        },
        {
            'rate': -1.0,
        }
    ]),
    Example(RelaxedBernoulli, [
        {
            'temperature': torch.tensor([1.5], requires_grad=True),
            'probs': torch.tensor([1.7, 0.2, 0.4], requires_grad=True),
        },
        {
            'temperature': torch.tensor([2.0]),
            'probs': torch.tensor([-1.0]),
        }
    ]),
    Example(RelaxedOneHotCategorical, [
        {
            'temperature': torch.tensor([0.5], requires_grad=True),
            'probs': torch.tensor([[-0.1, 0.2, 0.7], [0.5, 0.3, 0.2]], requires_grad=True)
        },
        {
            'temperature': torch.tensor([2.0]),
            'probs': torch.tensor([[-1.0, 0.0], [-1.0, 1.1]])
        }
    ]),
    Example(TransformedDistribution, [
        {
            'base_distribution': Normal(0, 1),
            'transforms': lambda x: x,
        },
        {
            'base_distribution': Normal(0, 1),
            'transforms': [lambda x: x],
        },
    ]),
    Example(Uniform, [
        {
            'low': torch.tensor([2.0], requires_grad=True),
            'high': torch.tensor([2.0], requires_grad=True),
        },
        {
            'low': torch.tensor([0.0], requires_grad=True),
            'high': torch.tensor([0.0], requires_grad=True),
        },
        {
            'low': torch.tensor([1.0], requires_grad=True),
            'high': torch.tensor([0.0], requires_grad=True),
        }
    ]),
    Example(Weibull, [
        {
            'scale': torch.tensor([0.0], requires_grad=True),
            'concentration': torch.tensor([0.0], requires_grad=True)
        },
        {
            'scale': torch.tensor([1.0], requires_grad=True),
            'concentration': torch.tensor([-1.0], requires_grad=True)
        }
    ]),
    Example(Wishart, [
        {
            'covariance_matrix': torch.tensor([[1.0, 0.0], [0.0, -2.0]], requires_grad=True),
            'df': torch.tensor([1.5], requires_grad=True),
        },
        {
            'covariance_matrix': torch.tensor([[1.0, 1.0], [1.0, -2.0]], requires_grad=True),
            'df': torch.tensor([3.], requires_grad=True),
        },
        {
            'covariance_matrix': torch.tensor([[1.0, 1.0], [1.0, -2.0]], requires_grad=True),
            'df': 3.,
        },
    ]),
    Example(ContinuousBernoulli, [
        {'probs': torch.tensor([1.1, 0.2, 0.4], requires_grad=True)},
        {'probs': torch.tensor([-0.5], requires_grad=True)},
        {'probs': 1.00001},
    ])
]


class DistributionsTestCase(TestCase):
    def setUp(self):
        """The tests assume that the validation flag is set."""
        torch.distributions.Distribution.set_default_validate_args(True)
        super(DistributionsTestCase, self).setUp()


class TestDistributions(DistributionsTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    def _gradcheck_log_prob(self, dist_ctor, ctor_params):
        # performs gradient checks on log_prob
        distribution = dist_ctor(*ctor_params)
        s = distribution.sample()
        if not distribution.support.is_discrete:
            s = s.detach().requires_grad_()

        expected_shape = distribution.batch_shape + distribution.event_shape
        self.assertEqual(s.size(), expected_shape)

        def apply_fn(s, *params):
            return dist_ctor(*params).log_prob(s)

        gradcheck(apply_fn, (s,) + tuple(ctor_params), raise_exception=True)

    def _check_forward_ad(self, fn):
        with fwAD.dual_level():
            x = torch.tensor(1.)
            t = torch.tensor(1.)
            dual = fwAD.make_dual(x, t)
            dual_out = fn(dual)
            self.assertEqual(torch.count_nonzero(fwAD.unpack_dual(dual_out).tangent).item(), 0)

    def _check_log_prob(self, dist, asset_fn):
        # checks that the log_prob matches a reference function
        s = dist.sample()
        log_probs = dist.log_prob(s)
        log_probs_data_flat = log_probs.view(-1)
        s_data_flat = s.view(len(log_probs_data_flat), -1)
        for i, (val, log_prob) in enumerate(zip(s_data_flat, log_probs_data_flat)):
            asset_fn(i, val.squeeze(), log_prob)

    def _check_sampler_sampler(self, torch_dist, ref_dist, message, multivariate=False,
                               circular=False, num_samples=10000, failure_rate=1e-3):
        # Checks that the .sample() method matches a reference function.
        torch_samples = torch_dist.sample((num_samples,)).squeeze()
        torch_samples = torch_samples.cpu().numpy()
        ref_samples = ref_dist.rvs(num_samples).astype(np.float64)
        if multivariate:
            # Project onto a random axis.
            axis = np.random.normal(size=(1,) + torch_samples.shape[1:])
            axis /= np.linalg.norm(axis)
            torch_samples = (axis * torch_samples).reshape(num_samples, -1).sum(-1)
            ref_samples = (axis * ref_samples).reshape(num_samples, -1).sum(-1)
        samples = [(x, +1) for x in torch_samples] + [(x, -1) for x in ref_samples]
        if circular:
            samples = [(np.cos(x), v) for (x, v) in samples]
        shuffle(samples)  # necessary to prevent stable sort from making uneven bins for discrete
        samples.sort(key=lambda x: x[0])
        samples = np.array(samples)[:, 1]

        # Aggregate into bins filled with roughly zero-mean unit-variance RVs.
        num_bins = 10
        samples_per_bin = len(samples) // num_bins
        bins = samples.reshape((num_bins, samples_per_bin)).mean(axis=1)
        stddev = samples_per_bin ** -0.5
        threshold = stddev * scipy.special.erfinv(1 - 2 * failure_rate / num_bins)
        message = '{}.sample() is biased:\n{}'.format(message, bins)
        for bias in bins:
            self.assertLess(-threshold, bias, message)
            self.assertLess(bias, threshold, message)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def _check_sampler_discrete(self, torch_dist, ref_dist, message,
                                num_samples=10000, failure_rate=1e-3):
        """Runs a Chi2-test for the support, but ignores tail instead of combining"""
        torch_samples = torch_dist.sample((num_samples,)).squeeze()
        torch_samples = torch_samples.cpu().numpy()
        unique, counts = np.unique(torch_samples, return_counts=True)
        pmf = ref_dist.pmf(unique)
        pmf = pmf / pmf.sum()  # renormalize to 1.0 for chisq test
        msk = (counts > 5) & ((pmf * num_samples) > 5)
        self.assertGreater(pmf[msk].sum(), 0.9, "Distribution is too sparse for test; try increasing num_samples")
        # Add a remainder bucket that combines counts for all values
        # below threshold, if such values exist (i.e. mask has False entries).
        if not msk.all():
            counts = np.concatenate([counts[msk], np.sum(counts[~msk], keepdims=True)])
            pmf = np.concatenate([pmf[msk], np.sum(pmf[~msk], keepdims=True)])
        chisq, p = scipy.stats.chisquare(counts, pmf * num_samples)
        self.assertGreater(p, failure_rate, message)

    def _check_enumerate_support(self, dist, examples):
        for params, expected in examples:
            params = {k: torch.tensor(v) for k, v in params.items()}
            d = dist(**params)
            actual = d.enumerate_support(expand=False)
            expected = torch.tensor(expected, dtype=actual.dtype)
            self.assertEqual(actual, expected)
            actual = d.enumerate_support(expand=True)
            expected_with_expand = expected.expand((-1,) + d.batch_shape + d.event_shape)
            self.assertEqual(actual, expected_with_expand)

    def test_lowrank_multivariate_normal_shape(self):
        mean = torch.randn(5, 3, requires_grad=True)
        mean_no_batch = torch.randn(3, requires_grad=True)
        mean_multi_batch = torch.randn(6, 5, 3, requires_grad=True)

        # construct PSD covariance
        cov_factor = torch.randn(3, 1, requires_grad=True)
        cov_diag = torch.randn(3).abs().requires_grad_()

        # construct batch of PSD covariances
        cov_factor_batched = torch.randn(6, 5, 3, 2, requires_grad=True)
        cov_diag_batched = torch.randn(6, 5, 3).abs().requires_grad_()

        # ensure that sample, batch, event shapes all handled correctly
        self.assertEqual(LowRankMultivariateNormal(mean, cov_factor, cov_diag)
                         .sample().size(), (5, 3))
        self.assertEqual(LowRankMultivariateNormal(mean_no_batch, cov_factor, cov_diag)
                         .sample().size(), (3,))
        self.assertEqual(LowRankMultivariateNormal(mean_multi_batch, cov_factor, cov_diag)
                         .sample().size(), (6, 5, 3))
        self.assertEqual(LowRankMultivariateNormal(mean, cov_factor, cov_diag)
                         .sample((2,)).size(), (2, 5, 3))
        self.assertEqual(LowRankMultivariateNormal(mean_no_batch, cov_factor, cov_diag)
                         .sample((2,)).size(), (2, 3))
        self.assertEqual(LowRankMultivariateNormal(mean_multi_batch, cov_factor, cov_diag)
                         .sample((2,)).size(), (2, 6, 5, 3))
        self.assertEqual(LowRankMultivariateNormal(mean, cov_factor, cov_diag)
                         .sample((2, 7)).size(), (2, 7, 5, 3))
        self.assertEqual(LowRankMultivariateNormal(mean_no_batch, cov_factor, cov_diag)
                         .sample((2, 7)).size(), (2, 7, 3))
        self.assertEqual(LowRankMultivariateNormal(mean_multi_batch, cov_factor, cov_diag)
                         .sample((2, 7)).size(), (2, 7, 6, 5, 3))
        self.assertEqual(LowRankMultivariateNormal(mean, cov_factor_batched, cov_diag_batched)
                         .sample((2, 7)).size(), (2, 7, 6, 5, 3))
        self.assertEqual(LowRankMultivariateNormal(mean_no_batch, cov_factor_batched, cov_diag_batched)
                         .sample((2, 7)).size(), (2, 7, 6, 5, 3))
        self.assertEqual(LowRankMultivariateNormal(mean_multi_batch, cov_factor_batched, cov_diag_batched)
                         .sample((2, 7)).size(), (2, 7, 6, 5, 3))

        # check gradients
        self._gradcheck_log_prob(LowRankMultivariateNormal,
                                 (mean, cov_factor, cov_diag))
        self._gradcheck_log_prob(LowRankMultivariateNormal,
                                 (mean_multi_batch, cov_factor, cov_diag))
        self._gradcheck_log_prob(LowRankMultivariateNormal,
                                 (mean_multi_batch, cov_factor_batched, cov_diag_batched))

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_lowrank_multivariate_normal_log_prob(self):
        mean = torch.randn(3, requires_grad=True)
        cov_factor = torch.randn(3, 1, requires_grad=True)
        cov_diag = torch.randn(3).abs().requires_grad_()
        cov = cov_factor.matmul(cov_factor.t()) + cov_diag.diag()

        # check that logprob values match scipy logpdf,
        # and that covariance and scale_tril parameters are equivalent
        dist1 = LowRankMultivariateNormal(mean, cov_factor, cov_diag)
        ref_dist = scipy.stats.multivariate_normal(mean.detach().numpy(), cov.detach().numpy())

        x = dist1.sample((10,))
        expected = ref_dist.logpdf(x.numpy())

        self.assertEqual(0.0, np.mean((dist1.log_prob(x).detach().numpy() - expected)**2), atol=1e-3, rtol=0)

        # Double-check that batched versions behave the same as unbatched
        mean = torch.randn(5, 3, requires_grad=True)
        cov_factor = torch.randn(5, 3, 2, requires_grad=True)
        cov_diag = torch.randn(5, 3).abs().requires_grad_()

        dist_batched = LowRankMultivariateNormal(mean, cov_factor, cov_diag)
        dist_unbatched = [LowRankMultivariateNormal(mean[i], cov_factor[i], cov_diag[i])
                          for i in range(mean.size(0))]

        x = dist_batched.sample((10,))
        batched_prob = dist_batched.log_prob(x)
        unbatched_prob = torch.stack([dist_unbatched[i].log_prob(x[:, i]) for i in range(5)]).t()

        self.assertEqual(batched_prob.shape, unbatched_prob.shape)
        self.assertEqual(0.0, (batched_prob - unbatched_prob).abs().max(), atol=1e-3, rtol=0)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_lowrank_multivariate_normal_sample(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        mean = torch.randn(5, requires_grad=True)
        cov_factor = torch.randn(5, 1, requires_grad=True)
        cov_diag = torch.randn(5).abs().requires_grad_()
        cov = cov_factor.matmul(cov_factor.t()) + cov_diag.diag()

        self._check_sampler_sampler(LowRankMultivariateNormal(mean, cov_factor, cov_diag),
                                    scipy.stats.multivariate_normal(mean.detach().numpy(), cov.detach().numpy()),
                                    'LowRankMultivariateNormal(loc={}, cov_factor={}, cov_diag={})'
                                    .format(mean, cov_factor, cov_diag), multivariate=True)

    def test_lowrank_multivariate_normal_properties(self):
        loc = torch.randn(5)
        cov_factor = torch.randn(5, 2)
        cov_diag = torch.randn(5).abs()
        cov = cov_factor.matmul(cov_factor.t()) + cov_diag.diag()
        m1 = LowRankMultivariateNormal(loc, cov_factor, cov_diag)
        m2 = MultivariateNormal(loc=loc, covariance_matrix=cov)
        self.assertEqual(m1.mean, m2.mean)
        self.assertEqual(m1.variance, m2.variance)
        self.assertEqual(m1.covariance_matrix, m2.covariance_matrix)
        self.assertEqual(m1.scale_tril, m2.scale_tril)
        self.assertEqual(m1.precision_matrix, m2.precision_matrix)
        self.assertEqual(m1.entropy(), m2.entropy())

    def test_lowrank_multivariate_normal_moments(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        mean = torch.randn(5)
        cov_factor = torch.randn(5, 2)
        cov_diag = torch.randn(5).abs()
        d = LowRankMultivariateNormal(mean, cov_factor, cov_diag)
        samples = d.rsample((100000,))
        empirical_mean = samples.mean(0)
        self.assertEqual(d.mean, empirical_mean, atol=0.01, rtol=0)
        empirical_var = samples.var(0)
        self.assertEqual(d.variance, empirical_var, atol=0.02, rtol=0)


class TestKL(DistributionsTestCase):

    def setUp(self):
        super(TestKL, self).setUp()

        class Binomial30(Binomial):
            def __init__(self, probs):
                super(Binomial30, self).__init__(30, probs)

        # These are pairs of distributions with 4 x 4 parameters as specified.
        # The first of the pair e.g. bernoulli[0] varies column-wise and the second
        # e.g. bernoulli[1] varies row-wise; that way we test all param pairs.
        bernoulli = pairwise(Bernoulli, [0.1, 0.2, 0.6, 0.9])
        binomial30 = pairwise(Binomial30, [0.1, 0.2, 0.6, 0.9])
        binomial_vectorized_count = (Binomial(torch.tensor([3, 4]), torch.tensor([0.4, 0.6])),
                                     Binomial(torch.tensor([3, 4]), torch.tensor([0.5, 0.8])))
        beta = pairwise(Beta, [1.0, 2.5, 1.0, 2.5], [1.5, 1.5, 3.5, 3.5])
        categorical = pairwise(Categorical, [[0.4, 0.3, 0.3],
                                             [0.2, 0.7, 0.1],
                                             [0.33, 0.33, 0.34],
                                             [0.2, 0.2, 0.6]])
        cauchy = pairwise(Cauchy, [-2.0, 2.0, -3.0, 3.0], [1.0, 2.0, 1.0, 2.0])
        chi2 = pairwise(Chi2, [1.0, 2.0, 2.5, 5.0])
        dirichlet = pairwise(Dirichlet, [[0.1, 0.2, 0.7],
                                         [0.5, 0.4, 0.1],
                                         [0.33, 0.33, 0.34],
                                         [0.2, 0.2, 0.4]])
        exponential = pairwise(Exponential, [1.0, 2.5, 5.0, 10.0])
        gamma = pairwise(Gamma, [1.0, 2.5, 1.0, 2.5], [1.5, 1.5, 3.5, 3.5])
        gumbel = pairwise(Gumbel, [-2.0, 4.0, -3.0, 6.0], [1.0, 2.5, 1.0, 2.5])
        halfnormal = pairwise(HalfNormal, [1.0, 2.0, 1.0, 2.0])
        laplace = pairwise(Laplace, [-2.0, 4.0, -3.0, 6.0], [1.0, 2.5, 1.0, 2.5])
        lognormal = pairwise(LogNormal, [-2.0, 2.0, -3.0, 3.0], [1.0, 2.0, 1.0, 2.0])
        normal = pairwise(Normal, [-2.0, 2.0, -3.0, 3.0], [1.0, 2.0, 1.0, 2.0])
        independent = (Independent(normal[0], 1), Independent(normal[1], 1))
        onehotcategorical = pairwise(OneHotCategorical, [[0.4, 0.3, 0.3],
                                                         [0.2, 0.7, 0.1],
                                                         [0.33, 0.33, 0.34],
                                                         [0.2, 0.2, 0.6]])
        pareto = (Pareto(torch.tensor([2.5, 4.0, 2.5, 4.0]).expand(4, 4),
                         torch.tensor([2.25, 3.75, 2.25, 3.75]).expand(4, 4)),
                  Pareto(torch.tensor([2.25, 3.75, 2.25, 3.8]).expand(4, 4),
                         torch.tensor([2.25, 3.75, 2.25, 3.75]).expand(4, 4)))
        poisson = pairwise(Poisson, [0.3, 1.0, 5.0, 10.0])
        uniform_within_unit = pairwise(Uniform, [0.1, 0.9, 0.2, 0.75], [0.15, 0.95, 0.25, 0.8])
        uniform_positive = pairwise(Uniform, [1, 1.5, 2, 4], [1.2, 2.0, 3, 7])
        uniform_real = pairwise(Uniform, [-2., -1, 0, 2], [-1., 1, 1, 4])
        uniform_pareto = pairwise(Uniform, [6.5, 7.5, 6.5, 8.5], [7.5, 8.5, 9.5, 9.5])
        continuous_bernoulli = pairwise(ContinuousBernoulli, [0.1, 0.2, 0.5, 0.9])

        # These tests should pass with precision = 0.01, but that makes tests very expensive.
        # Instead, we test with precision = 0.1 and only test with higher precision locally
        # when adding a new KL implementation.
        # The following pairs are not tested due to very high variance of the monte carlo
        # estimator; their implementations have been reviewed with extra care:
        # - (pareto, normal)
        self.precision = 0.1  # Set this to 0.01 when testing a new KL implementation.
        self.max_samples = int(1e07)  # Increase this when testing at smaller precision.
        self.samples_per_batch = int(1e04)
        self.finite_examples = [
            (bernoulli, bernoulli),
            (bernoulli, poisson),
            (beta, beta),
            (beta, chi2),
            (beta, exponential),
            (beta, gamma),
            (beta, normal),
            (binomial30, binomial30),
            (binomial_vectorized_count, binomial_vectorized_count),
            (categorical, categorical),
            (cauchy, cauchy),
            (chi2, chi2),
            (chi2, exponential),
            (chi2, gamma),
            (chi2, normal),
            (dirichlet, dirichlet),
            (exponential, chi2),
            (exponential, exponential),
            (exponential, gamma),
            (exponential, gumbel),
            (exponential, normal),
            (gamma, chi2),
            (gamma, exponential),
            (gamma, gamma),
            (gamma, gumbel),
            (gamma, normal),
            (gumbel, gumbel),
            (gumbel, normal),
            (halfnormal, halfnormal),
            (independent, independent),
            (laplace, laplace),
            (lognormal, lognormal),
            (laplace, normal),
            (normal, gumbel),
            (normal, laplace),
            (normal, normal),
            (onehotcategorical, onehotcategorical),
            (pareto, chi2),
            (pareto, pareto),
            (pareto, exponential),
            (pareto, gamma),
            (poisson, poisson),
            (uniform_within_unit, beta),
            (uniform_positive, chi2),
            (uniform_positive, exponential),
            (uniform_positive, gamma),
            (uniform_real, gumbel),
            (uniform_real, normal),
            (uniform_pareto, pareto),
            (continuous_bernoulli, continuous_bernoulli),
            (continuous_bernoulli, exponential),
            (continuous_bernoulli, normal),
            (beta, continuous_bernoulli)
        ]

        self.infinite_examples = [
            (Bernoulli(0), Bernoulli(1)),
            (Bernoulli(1), Bernoulli(0)),
            (Categorical(torch.tensor([0.9, 0.1])), Categorical(torch.tensor([1., 0.]))),
            (Categorical(torch.tensor([[0.9, 0.1], [.9, .1]])), Categorical(torch.tensor([1., 0.]))),
            (Beta(1, 2), Uniform(0.25, 1)),
            (Beta(1, 2), Uniform(0, 0.75)),
            (Beta(1, 2), Uniform(0.25, 0.75)),
            (Beta(1, 2), Pareto(1, 2)),
            (Binomial(31, 0.7), Binomial(30, 0.3)),
            (Binomial(torch.tensor([3, 4]), torch.tensor([0.4, 0.6])),
             Binomial(torch.tensor([2, 3]), torch.tensor([0.5, 0.8]))),
            (Chi2(1), Beta(2, 3)),
            (Chi2(1), Pareto(2, 3)),
            (Chi2(1), Uniform(-2, 3)),
            (Exponential(1), Beta(2, 3)),
            (Exponential(1), Pareto(2, 3)),
            (Exponential(1), Uniform(-2, 3)),
            (Gamma(1, 2), Beta(3, 4)),
            (Gamma(1, 2), Pareto(3, 4)),
            (Gamma(1, 2), Uniform(-3, 4)),
            (Gumbel(-1, 2), Beta(3, 4)),
            (Gumbel(-1, 2), Chi2(3)),
            (Gumbel(-1, 2), Exponential(3)),
            (Gumbel(-1, 2), Gamma(3, 4)),
            (Gumbel(-1, 2), Pareto(3, 4)),
            (Gumbel(-1, 2), Uniform(-3, 4)),
            (Laplace(-1, 2), Beta(3, 4)),
            (Laplace(-1, 2), Chi2(3)),
            (Laplace(-1, 2), Exponential(3)),
            (Laplace(-1, 2), Gamma(3, 4)),
            (Laplace(-1, 2), Pareto(3, 4)),
            (Laplace(-1, 2), Uniform(-3, 4)),
            (Normal(-1, 2), Beta(3, 4)),
            (Normal(-1, 2), Chi2(3)),
            (Normal(-1, 2), Exponential(3)),
            (Normal(-1, 2), Gamma(3, 4)),
            (Normal(-1, 2), Pareto(3, 4)),
            (Normal(-1, 2), Uniform(-3, 4)),
            (Pareto(2, 1), Chi2(3)),
            (Pareto(2, 1), Exponential(3)),
            (Pareto(2, 1), Gamma(3, 4)),
            (Pareto(1, 2), Normal(-3, 4)),
            (Pareto(1, 2), Pareto(3, 4)),
            (Poisson(2), Bernoulli(0.5)),
            (Poisson(2.3), Binomial(10, 0.2)),
            (Uniform(-1, 1), Beta(2, 2)),
            (Uniform(0, 2), Beta(3, 4)),
            (Uniform(-1, 2), Beta(3, 4)),
            (Uniform(-1, 2), Chi2(3)),
            (Uniform(-1, 2), Exponential(3)),
            (Uniform(-1, 2), Gamma(3, 4)),
            (Uniform(-1, 2), Pareto(3, 4)),
            (ContinuousBernoulli(0.25), Uniform(0.25, 1)),
            (ContinuousBernoulli(0.25), Uniform(0, 0.75)),
            (ContinuousBernoulli(0.25), Uniform(0.25, 0.75)),
            (ContinuousBernoulli(0.25), Pareto(1, 2)),
            (Exponential(1), ContinuousBernoulli(0.75)),
            (Gamma(1, 2), ContinuousBernoulli(0.75)),
            (Gumbel(-1, 2), ContinuousBernoulli(0.75)),
            (Laplace(-1, 2), ContinuousBernoulli(0.75)),
            (Normal(-1, 2), ContinuousBernoulli(0.75)),
            (Uniform(-1, 1), ContinuousBernoulli(0.75)),
            (Uniform(0, 2), ContinuousBernoulli(0.75)),
            (Uniform(-1, 2), ContinuousBernoulli(0.75))
        ]


    def test_kl_lowrank_multivariate_normal(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        n = 5  # Number of tests for lowrank_multivariate_normal
        for i in range(0, n):
            loc = [torch.randn(4) for _ in range(0, 2)]
            cov_factor = [torch.randn(4, 3) for _ in range(0, 2)]
            cov_diag = [transform_to(constraints.positive)(torch.randn(4)) for _ in range(0, 2)]
            covariance_matrix = [cov_factor[i].matmul(cov_factor[i].t()) +
                                 cov_diag[i].diag() for i in range(0, 2)]
            p = LowRankMultivariateNormal(loc[0], cov_factor[0], cov_diag[0])
            q = LowRankMultivariateNormal(loc[1], cov_factor[1], cov_diag[1])
            p_full = MultivariateNormal(loc[0], covariance_matrix[0])
            q_full = MultivariateNormal(loc[1], covariance_matrix[1])
            expected = kl_divergence(p_full, q_full)

            actual_lowrank_lowrank = kl_divergence(p, q)
            actual_lowrank_full = kl_divergence(p, q_full)
            actual_full_lowrank = kl_divergence(p_full, q)

            error_lowrank_lowrank = torch.abs(actual_lowrank_lowrank - expected).max()
            self.assertLess(error_lowrank_lowrank, self.precision, '\n'.join([
                'Incorrect KL(LowRankMultivariateNormal, LowRankMultivariateNormal) instance {}/{}'.format(i + 1, n),
                'Expected (from KL MultivariateNormal): {}'.format(expected),
                'Actual (analytic): {}'.format(actual_lowrank_lowrank),
            ]))

            error_lowrank_full = torch.abs(actual_lowrank_full - expected).max()
            self.assertLess(error_lowrank_full, self.precision, '\n'.join([
                'Incorrect KL(LowRankMultivariateNormal, MultivariateNormal) instance {}/{}'.format(i + 1, n),
                'Expected (from KL MultivariateNormal): {}'.format(expected),
                'Actual (analytic): {}'.format(actual_lowrank_full),
            ]))

            error_full_lowrank = torch.abs(actual_full_lowrank - expected).max()
            self.assertLess(error_full_lowrank, self.precision, '\n'.join([
                'Incorrect KL(MultivariateNormal, LowRankMultivariateNormal) instance {}/{}'.format(i + 1, n),
                'Expected (from KL MultivariateNormal): {}'.format(expected),
                'Actual (analytic): {}'.format(actual_full_lowrank),
            ]))

    def test_kl_lowrank_multivariate_normal_batched(self):
        b = 7  # Number of batches
        loc = [torch.randn(b, 3) for _ in range(0, 2)]
        cov_factor = [torch.randn(b, 3, 2) for _ in range(0, 2)]
        cov_diag = [transform_to(constraints.positive)(torch.randn(b, 3)) for _ in range(0, 2)]
        expected_kl = torch.stack([
            kl_divergence(LowRankMultivariateNormal(loc[0][i], cov_factor[0][i], cov_diag[0][i]),
                          LowRankMultivariateNormal(loc[1][i], cov_factor[1][i], cov_diag[1][i]))
            for i in range(0, b)])
        actual_kl = kl_divergence(LowRankMultivariateNormal(loc[0], cov_factor[0], cov_diag[0]),
                                  LowRankMultivariateNormal(loc[1], cov_factor[1], cov_diag[1]))
        self.assertEqual(expected_kl, actual_kl)

    
if __name__ == '__main__' and torch._C.has_lapack:
    run_tests()
