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
    ])
]

BAD_EXAMPLES = [
    Example(RelaxedOneHotCategorical, [
        {
            'temperature': torch.tensor([0.5], requires_grad=True),
            'probs': torch.tensor([[-0.1, 0.2, 0.7], [0.5, 0.3, 0.2]], requires_grad=True)
        },
        {
            'temperature': torch.tensor([2.0]),
            'probs': torch.tensor([[-1.0, 0.0], [-1.0, 1.1]])
        }
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
