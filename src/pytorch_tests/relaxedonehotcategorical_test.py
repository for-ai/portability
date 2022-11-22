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

    def test_relaxed_one_hot_categorical_1d(self):
        p = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
        temp = torch.tensor(0.67, requires_grad=True)
        self.assertEqual(RelaxedOneHotCategorical(probs=p, temperature=temp).sample().size(), (3,))
        self.assertFalse(RelaxedOneHotCategorical(probs=p, temperature=temp).sample().requires_grad)
        self.assertEqual(RelaxedOneHotCategorical(probs=p, temperature=temp).sample((2, 2)).size(), (2, 2, 3))
        self.assertEqual(RelaxedOneHotCategorical(probs=p, temperature=temp).sample((1,)).size(), (1, 3))
        self._gradcheck_log_prob(lambda t, p: RelaxedOneHotCategorical(t, p, validate_args=False), (temp, p))

    def test_relaxed_one_hot_categorical_2d(self):
        probabilities = [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]]
        probabilities_1 = [[1.0, 0.0], [0.0, 1.0]]
        temp = torch.tensor([3.0], requires_grad=True)
        # The lower the temperature, the more unstable the log_prob gradcheck is
        # w.r.t. the sample. Values below 0.25 empirically fail the default tol.
        temp_2 = torch.tensor([0.25], requires_grad=True)
        p = torch.tensor(probabilities, requires_grad=True)
        s = torch.tensor(probabilities_1, requires_grad=True)
        self.assertEqual(RelaxedOneHotCategorical(temp, p).sample().size(), (2, 3))
        self.assertEqual(RelaxedOneHotCategorical(temp, p).sample(sample_shape=(3, 4)).size(), (3, 4, 2, 3))
        self.assertEqual(RelaxedOneHotCategorical(temp, p).sample((6,)).size(), (6, 2, 3))
        self._gradcheck_log_prob(lambda t, p: RelaxedOneHotCategorical(t, p, validate_args=False), (temp, p))
        self._gradcheck_log_prob(lambda t, p: RelaxedOneHotCategorical(t, p, validate_args=False), (temp_2, p))

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_argmax_relaxed_categorical(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]

        class ArgMax(object):
            def __init__(self, dist):
                self.dist = dist

            def sample(self, *args, **kwargs):
                s = self.dist.sample(*args, **kwargs)
                _, idx = torch.max(s, -1)
                return idx

        class ScipyCategorical(object):
            def __init__(self, dist):
                self.dist = dist

            def pmf(self, samples):
                new_samples = np.zeros(samples.shape + self.dist.p.shape)
                new_samples[np.arange(samples.shape[0]), samples] = 1
                return self.dist.pmf(new_samples)

        for probs, temp in product([torch.tensor([0.1, 0.9]), torch.tensor([0.2, 0.2, 0.6])], [0.1, 1.0, 10.0]):
            self._check_sampler_discrete(ArgMax(RelaxedOneHotCategorical(temp, probs)),
                                         ScipyCategorical(scipy.stats.multinomial(1, probs)),
                                         'Rounded(RelaxedOneHotCategorical(temp={}, probs={}))'.format(temp, probs),
                                         failure_rate=1e-3)

        for probs in [torch.tensor([0.1, 0.9]), torch.tensor([0.2, 0.2, 0.6])]:
            equal_probs = torch.ones(probs.size()) / probs.size()[0]
            dist = RelaxedOneHotCategorical(1e10, probs)
            s = dist.rsample()
            self.assertEqual(equal_probs, s)

    def _test_discrete_distribution_mode(self, dist, sanitized_mode, batch_isfinite):
        # We cannot easily check the mode for discrete distributions, but we can look left and right
        # to ensure the log probability is smaller than at the mode.
        for step in [-1, 1]:
            log_prob_mode = dist.log_prob(sanitized_mode)
            if isinstance(dist, OneHotCategorical):
                idx = (dist._categorical.mode + 1) % dist.probs.shape[-1]
                other = torch.nn.functional.one_hot(idx, num_classes=dist.probs.shape[-1]).to(dist.mode)
            else:
                other = dist.mode + step
            mask = batch_isfinite & dist.support.check(other)
            self.assertTrue(mask.any() or dist.mode.unique().numel() == 1)
            # Add a dimension to the right if the event shape is not a scalar, e.g. OneHotCategorical.
            other = torch.where(mask[..., None] if mask.ndim < other.ndim else mask, other, dist.sample())
            log_prob_other = dist.log_prob(other)
            delta = log_prob_mode - log_prob_other
            self.assertTrue((-1e-12 < delta[mask].detach()).all())  # Allow up to 1e-12 rounding error.

    def _test_continuous_distribution_mode(self, dist, sanitized_mode, batch_isfinite):
        if isinstance(dist, Wishart):
            return
        # We perturb the mode in the unconstrained space and expect the log probability to decrease.
        num_points = 10
        transform = transform_to(dist.support)
        unconstrained_mode = transform.inv(sanitized_mode)
        perturbation = 1e-5 * (torch.rand((num_points,) + unconstrained_mode.shape) - 0.5)
        perturbed_mode = transform(perturbation + unconstrained_mode)
        log_prob_mode = dist.log_prob(sanitized_mode)
        log_prob_other = dist.log_prob(perturbed_mode)
        delta = log_prob_mode - log_prob_other

        # We pass the test with a small tolerance to allow for rounding and manually set the
        # difference to zero if both log probs are infinite with the same sign.
        both_infinite_with_same_sign = (log_prob_mode == log_prob_other) & (log_prob_mode.abs() == inf)
        delta[both_infinite_with_same_sign] = 0.
        ordering = (delta > -1e-12).all(axis=0)
        self.assertTrue(ordering[batch_isfinite].all())

    def test_mode(self):
        discrete_distributions = (
            Bernoulli, Binomial, Categorical, Geometric, NegativeBinomial, OneHotCategorical, Poisson,
        )
        no_mode_available = (
            ContinuousBernoulli, LKJCholesky, LogisticNormal, MixtureSameFamily, Multinomial,
            RelaxedBernoulli, RelaxedOneHotCategorical,
        )

        for dist_cls, params in EXAMPLES:
            for param in params:
                dist = dist_cls(**param)
                if isinstance(dist, no_mode_available) or type(dist) is TransformedDistribution:
                    with self.assertRaises(NotImplementedError):
                        dist.mode
                    continue

                # Check that either all or no elements in the event shape are nan: the mode cannot be
                # defined for part of an event.
                isfinite = dist.mode.isfinite().reshape(dist.batch_shape + (dist.event_shape.numel(),))
                batch_isfinite = isfinite.all(axis=-1)
                self.assertTrue((batch_isfinite | ~isfinite.any(axis=-1)).all())

                # We sanitize undefined modes by sampling from the distribution.
                sanitized_mode = torch.where(~dist.mode.isnan(), dist.mode, dist.sample())
                if isinstance(dist, discrete_distributions):
                    self._test_discrete_distribution_mode(dist, sanitized_mode, batch_isfinite)
                else:
                    self._test_continuous_distribution_mode(dist, sanitized_mode, batch_isfinite)

                self.assertFalse(dist.log_prob(sanitized_mode).isnan().any())



class TestConstraints(DistributionsTestCase):
    def test_params_constraints(self):
        normalize_probs_dists = (
            Categorical,
            Multinomial,
            OneHotCategorical,
            OneHotCategoricalStraightThrough,
            RelaxedOneHotCategorical
        )

        for Dist, params in EXAMPLES:
            for i, param in enumerate(params):
                dist = Dist(**param)
                for name, value in param.items():
                    if isinstance(value, numbers.Number):
                        value = torch.tensor([value])
                    if Dist in normalize_probs_dists and name == 'probs':
                        # These distributions accept positive probs, but elsewhere we
                        # use a stricter constraint to the simplex.
                        value = value / value.sum(-1, True)
                    try:
                        constraint = dist.arg_constraints[name]
                    except KeyError:
                        continue  # ignore optional parameters

                    # Check param shape is compatible with distribution shape.
                    self.assertGreaterEqual(value.dim(), constraint.event_dim)
                    value_batch_shape = value.shape[:value.dim() - constraint.event_dim]
                    torch.broadcast_shapes(dist.batch_shape, value_batch_shape)

                    if is_dependent(constraint):
                        continue

                    message = '{} example {}/{} parameter {} = {}'.format(
                        Dist.__name__, i + 1, len(params), name, value)
                    self.assertTrue(constraint.check(value).all(), msg=message)

if __name__ == '__main__' and torch._C.has_lapack:
    run_tests()
