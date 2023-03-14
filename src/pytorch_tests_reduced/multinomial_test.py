import math
import numbers
import unittest
from collections import namedtuple
from itertools import product
from random import shuffle
from packaging import version

import torch
from hypothesis import given
import torch.testing._internal.hypothesis_utils as hu

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
from ..utils.pytorch_device_decorators import onlyNativeDeviceTypes, onlyAcceleratedDeviceTypes, instantiate_device_type_tests
from ..utils.timer_wrapper import pytorch_op_timer

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
        with pytorch_op_timer():
            distribution = dist_ctor(*ctor_params)
        s = distribution.sample()
        if not distribution.support.is_discrete:
            s = s.detach().requires_grad_()
        expected_shape = distribution.batch_shape + distribution.event_shape
        self.assertEqual(s.size(), expected_shape)
        def apply_fn(s, *params):
            return dist_ctor(*params).log_prob(s)
        gradcheck(apply_fn, (s,) + tuple(ctor_params), raise_exception=True)
        
    def test_multinomial_1d(self, device):
        total_count = 10
        p = torch.tensor([0.1, 0.2, 0.3], requires_grad=True, device=device)
        with pytorch_op_timer():
            test_1 = Multinomial(total_count, p).sample().size()
        self.assertEqual(test_1, (3,))
        with pytorch_op_timer():
            test_2 = Multinomial(total_count, p).sample((2, 2)).size()
        self.assertEqual(test_2, (2, 2, 3))
        with pytorch_op_timer():
            test_3 = Multinomial(total_count, p).sample((1,)).size()
        self.assertEqual(test_3, (1, 3))
        self._gradcheck_log_prob(lambda p: Multinomial(total_count, p), [p])
        self._gradcheck_log_prob(lambda p: Multinomial(total_count, None, p.log()), [p])
        self.assertRaises(NotImplementedError, Multinomial(10, p).rsample)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_multinomial_1d_log_prob_and_entropy(self):
        total_count = 10
        p = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
        with pytorch_op_timer():
            dist = Multinomial(total_count, probs=p)
        x = dist.sample()
        log_prob = dist.log_prob(x)
        expected = torch.tensor(scipy.stats.multinomial.logpmf(x.numpy(), n=total_count, p=dist.probs.detach().numpy()))
        self.assertEqual(log_prob, expected)
        with pytorch_op_timer():
            dist = Multinomial(total_count, logits=p.log())
        x = dist.sample()
        log_prob = dist.log_prob(x)
        expected = torch.tensor(scipy.stats.multinomial.logpmf(x.numpy(), n=total_count, p=dist.probs.detach().numpy()))
        self.assertEqual(log_prob, expected)

        expected = scipy.stats.multinomial.entropy(total_count, dist.probs.detach().numpy())
        self.assertEqual(dist.entropy(), expected, atol=1e-3, rtol=0)

    def test_multinomial_2d(self, device):
        total_count = 10
        probabilities = [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]]
        probabilities_1 = [[1.0, 0.0], [0.0, 1.0]]
        p = torch.tensor(probabilities, requires_grad=True)
        s = torch.tensor(probabilities_1, requires_grad=True)
        with pytorch_op_timer():
            test_1 = Multinomial(total_count, p).sample().size()
        self.assertEqual(test_1, (2, 3))
        with pytorch_op_timer():   
            test_2 = Multinomial(total_count, p).sample(sample_shape=(3, 4)).size()         
        self.assertEqual(test_2, (3, 4, 2, 3))
        with pytorch_op_timer():
            test_3 = Multinomial(total_count, p).sample((6,)).size()
        self.assertEqual(test_3, (6, 2, 3))
        set_rng_seed(0)
        self._gradcheck_log_prob(lambda p: Multinomial(total_count, p), [p])
        self._gradcheck_log_prob(lambda p: Multinomial(total_count, None, p.log()), [p])

        # sample check for extreme value of probs
        self.assertEqual(Multinomial(total_count, s).sample(),
                        torch.tensor([[total_count, 0], [0, total_count]], dtype=torch.float64, device=device))


class TestDistributionShapes(DistributionsTestCase):
    def test_multinomial_shape(self, device):
        scalar_sample = 1
        tensor_sample_1 = torch.ones(3, 2, device=device)
        tensor_sample_2 = torch.ones(3, 2, 3, device=device)
        
        with pytorch_op_timer():
            dist = Multinomial(10, torch.tensor([[0.6, 0.3], [0.6, 0.3], [0.6, 0.3]], device=device))
        self.assertEqual(dist._batch_shape, torch.Size((3,)))
        self.assertEqual(dist._event_shape, torch.Size((2,)))
        self.assertEqual(dist.sample().size(), torch.Size((3, 2)))
        self.assertEqual(dist.sample((3, 2)).size(), torch.Size((3, 2, 3, 2)))
        self.assertEqual(dist.log_prob(tensor_sample_1).size(), torch.Size((3,)))
        self.assertRaises(ValueError, dist.log_prob, tensor_sample_2)
        self.assertEqual(dist.log_prob(torch.ones(3, 1, 2, device=device)).size(), torch.Size((3, 3)))


class TestNumericalStability(DistributionsTestCase):

    def test_multinomial_log_prob_with_logits(self, device):
        for dtype in ([torch.float, torch.double]):
            p = torch.tensor([-inf, 0], dtype=dtype, requires_grad=True, device=device)
            with pytorch_op_timer():
                multinomial = Multinomial(10, logits=p)
            log_pdf_prob_1 = multinomial.log_prob(torch.tensor([0, 10], dtype=dtype, device=device))
            self.assertEqual(log_pdf_prob_1.item(), 0)
            log_pdf_prob_0 = multinomial.log_prob(torch.tensor([10, 0], dtype=dtype, device=device))
            self.assertEqual(log_pdf_prob_0.item(), -inf)


instantiate_device_type_tests(TestNumericalStability, globals())
instantiate_device_type_tests(TestDistributions, globals())
instantiate_device_type_tests(TestDistributionShapes, globals())


if __name__ == '__main__' and torch._C.has_lapack:
    run_tests()
