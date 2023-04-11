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
from hypothesis import given
import torch.testing._internal.hypothesis_utils as hu

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


# Register all distributions for generic tests.
Example = namedtuple('Example', ['Dist', 'params'])
EXAMPLES = [
    Example(VonMises, [
        {
            'loc': torch.tensor(1.0, requires_grad=True),
            'concentration': torch.tensor(10.0, requires_grad=True)
        },
        {
            'loc': torch.tensor([0.0, math.pi / 2], requires_grad=True),
            'concentration': torch.tensor([1.0, 10.0], requires_grad=True)
        },
    ])
]


class DistributionsTestCase(TestCase):
    def setUp(self):
        """The tests assume that the validation flag is set."""
        torch.distributions.Distribution.set_default_validate_args(True)
        super(DistributionsTestCase, self).setUp()

# These tests are only needed for a few distributions that implement custom
# reparameterized gradients. Most .rsample() implementations simply rely on
# the reparameterization trick and do not need to be tested for accuracy.

class TestDistributionShapes(DistributionsTestCase):
    def setUp(self):
        # super(TestDistributionShapes, self).setUp()
        self.scalar_sample = 1
        self.tensor_sample_1 = torch.ones(3, 2)
        self.tensor_sample_2 = torch.ones(3, 2, 3)
        torch.set_default_dtype(torch.double)
    
    def tearDown(self):
        torch.set_default_dtype(torch.float)

    def test_vonmises_shape_tensor_params(self, device):
        with pytorch_op_timer():
            von_mises = VonMises(torch.tensor([0., 0.], device=device), torch.tensor([1., 1.], device=device))
        self.assertEqual(von_mises._batch_shape, torch.Size((2,)))
        self.assertEqual(von_mises._event_shape, torch.Size(()))
        self.assertEqual(von_mises.sample().size(), torch.Size((2,)))
        self.assertEqual(von_mises.sample(torch.Size((3, 2))).size(), torch.Size((3, 2, 2)))
        self.assertEqual(von_mises.log_prob(self.tensor_sample_1.to(device)).size(), torch.Size((3, 2)))
        self.assertEqual(von_mises.log_prob(torch.ones(2, 1, device=device)).size(), torch.Size((2, 2)))

    def test_vonmises_shape_scalar_params(self, device):
        with pytorch_op_timer():
            von_mises = VonMises(0., 1.)
        self.assertEqual(von_mises._batch_shape, torch.Size())
        self.assertEqual(von_mises._event_shape, torch.Size())
        self.assertEqual(von_mises.sample().size(), torch.Size())
        self.assertEqual(von_mises.sample(torch.Size((3, 2))).size(),
                         torch.Size((3, 2)))
        self.assertEqual(von_mises.log_prob(self.tensor_sample_1).size(),
                         torch.Size((3, 2)))
        self.assertEqual(von_mises.log_prob(self.tensor_sample_2).size(),
                         torch.Size((3, 2, 3)))

# instantiate_device_type_tests(TestRsample, globals())
instantiate_device_type_tests(TestDistributionShapes, globals())
if __name__ == '__main__' and torch._C.has_lapack:
    run_tests()
