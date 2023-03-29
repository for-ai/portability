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

from torch.distributions.transforms import (AffineTransform, ExpTransform)
from torch.distributions.constraints import Constraint, is_dependent
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
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import \
    (TestCase, run_tests)
import math
import numbers
from collections import namedtuple

import torch

# TODO: remove this global setting
# Distributions tests use double as the default dtype
torch.set_default_dtype(torch.double)

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
        {'probs': torch.tensor(
            [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True)},
        {'probs': torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)},
        {'logits': torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)},
    ]),
    Example(Binomial, [
        {'probs': torch.tensor(
            [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True), 'total_count': 10},
        {'probs': torch.tensor([[1.0, 0.0], [0.0, 1.0]],
                               requires_grad=True), 'total_count': 10},
        {'probs': torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]], requires_grad=True), 'total_count': torch.tensor([10])},
        {'probs': torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]], requires_grad=True), 'total_count': torch.tensor([10, 8])},
        {'probs': torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True),
         'total_count': torch.tensor([[10., 8.], [5., 3.]])},
        {'probs': torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True),
         'total_count': torch.tensor(0.)},
    ]),
    Example(NegativeBinomial, [
        {'probs': torch.tensor(
            [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True), 'total_count': 10},
        {'probs': torch.tensor([[0.9, 0.0], [0.0, 0.9]],
                               requires_grad=True), 'total_count': 10},
        {'probs': torch.tensor(
            [[0.9, 0.0], [0.0, 0.9]], requires_grad=True), 'total_count': torch.tensor([10])},
        {'probs': torch.tensor(
            [[0.9, 0.0], [0.0, 0.9]], requires_grad=True), 'total_count': torch.tensor([10, 8])},
        {'probs': torch.tensor([[0.9, 0.0], [0.0, 0.9]], requires_grad=True),
         'total_count': torch.tensor([[10., 8.], [5., 3.]])},
        {'probs': torch.tensor([[0.9, 0.0], [0.0, 0.9]], requires_grad=True),
         'total_count': torch.tensor(0.)},
    ]),
    Example(Multinomial, [
        {'probs': torch.tensor(
            [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True), 'total_count': 10},
        {'probs': torch.tensor([[1.0, 0.0], [0.0, 1.0]],
                               requires_grad=True), 'total_count': 10},
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
        {'probs': torch.tensor(
            [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True)},
        {'probs': torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)},
        {'logits': torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)},
    ]),
    Example(OneHotCategoricalStraightThrough, [
        {'probs': torch.tensor(
            [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True)},
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
        {'probs': torch.tensor(
            [[-0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True)},
        {'probs': torch.tensor(
            [[-1.0, 10.0], [0.0, -1.0]], requires_grad=True)},
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
        {'probs': torch.tensor(
            [[0.1, 0.2, 0.3], [0.1, -10.0, 0.2]], requires_grad=True)},
        {'probs': torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)},
    ]),
    Example(OneHotCategoricalStraightThrough, [
        {'probs': torch.tensor(
            [[0.1, 0.2, 0.3], [0.1, -10.0, 0.2]], requires_grad=True)},
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
                    value_batch_shape = value.shape[:value.dim(
                    ) - constraint.event_dim]
                    torch.broadcast_shapes(dist.batch_shape, value_batch_shape)

                    if is_dependent(constraint):
                        continue

                    message = '{} example {}/{} parameter {} = {}'.format(
                        Dist.__name__, i + 1, len(params), name, value)
                    self.assertTrue(constraint.check(value).all(), msg=message)

    def test_support_constraints(self):
        for Dist, params in EXAMPLES:
            self.assertIsInstance(Dist.support, Constraint)
            for i, param in enumerate(params):
                dist = Dist(**param)
                value = dist.sample()
                constraint = dist.support
                message = '{} example {}/{} sample = {}'.format(
                    Dist.__name__, i + 1, len(params), value)
                self.assertEqual(constraint.event_dim, len(
                    dist.event_shape), msg=message)
                ok = constraint.check(value)
                self.assertEqual(ok.shape, dist.batch_shape, msg=message)
                self.assertTrue(ok.all(), msg=message)


if __name__ == '__main__' and torch._C.has_lapack:
    run_tests()
