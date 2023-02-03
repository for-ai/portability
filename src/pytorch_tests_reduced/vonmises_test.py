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
from hypothesis import given
import torch.testing._internal.hypothesis_utils as hu

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
class TestRsample(DistributionsTestCase):
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_gamma(self):
        num_samples = 100
        for alpha in [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]:
            alphas = torch.tensor([alpha] * num_samples, dtype=torch.float, requires_grad=True)
            betas = alphas.new_ones(num_samples)
            x = Gamma(alphas, betas).rsample()
            x.sum().backward()
            x, ind = x.sort()
            x = x.detach().numpy()
            actual_grad = alphas.grad[ind].numpy()
            # Compare with expected gradient dx/dalpha along constant cdf(x,alpha).
            cdf = scipy.stats.gamma.cdf
            pdf = scipy.stats.gamma.pdf
            eps = 0.01 * alpha / (1.0 + alpha ** 0.5)
            cdf_alpha = (cdf(x, alpha + eps) - cdf(x, alpha - eps)) / (2 * eps)
            cdf_x = pdf(x, alpha)
            expected_grad = -cdf_alpha / cdf_x
            rel_error = np.abs(actual_grad - expected_grad) / (expected_grad + 1e-30)
            self.assertLess(np.max(rel_error), 0.0005, '\n'.join([
                'Bad gradient dx/alpha for x ~ Gamma({}, 1)'.format(alpha),
                'x {}'.format(x),
                'expected {}'.format(expected_grad),
                'actual {}'.format(actual_grad),
                'rel error {}'.format(rel_error),
                'max error {}'.format(rel_error.max()),
                'at alpha={}, x={}'.format(alpha, x[rel_error.argmax()]),
            ]))

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_chi2(self):
        num_samples = 100
        for df in [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]:
            dfs = torch.tensor([df] * num_samples, dtype=torch.float, requires_grad=True)
            x = Chi2(dfs).rsample()
            x.sum().backward()
            x, ind = x.sort()
            x = x.detach().numpy()
            actual_grad = dfs.grad[ind].numpy()
            # Compare with expected gradient dx/ddf along constant cdf(x,df).
            cdf = scipy.stats.chi2.cdf
            pdf = scipy.stats.chi2.pdf
            eps = 0.01 * df / (1.0 + df ** 0.5)
            cdf_df = (cdf(x, df + eps) - cdf(x, df - eps)) / (2 * eps)
            cdf_x = pdf(x, df)
            expected_grad = -cdf_df / cdf_x
            rel_error = np.abs(actual_grad - expected_grad) / (expected_grad + 1e-30)
            self.assertLess(np.max(rel_error), 0.001, '\n'.join([
                'Bad gradient dx/ddf for x ~ Chi2({})'.format(df),
                'x {}'.format(x),
                'expected {}'.format(expected_grad),
                'actual {}'.format(actual_grad),
                'rel error {}'.format(rel_error),
                'max error {}'.format(rel_error.max()),
            ]))

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_dirichlet_on_diagonal(self):
        num_samples = 20
        grid = [1e-1, 1e0, 1e1]
        for a0, a1, a2 in product(grid, grid, grid):
            alphas = torch.tensor([[a0, a1, a2]] * num_samples, dtype=torch.float, requires_grad=True)
            x = Dirichlet(alphas).rsample()[:, 0]
            x.sum().backward()
            x, ind = x.sort()
            x = x.detach().numpy()
            actual_grad = alphas.grad[ind].numpy()[:, 0]
            # Compare with expected gradient dx/dalpha0 along constant cdf(x,alpha).
            # This reduces to a distribution Beta(alpha[0], alpha[1] + alpha[2]).
            cdf = scipy.stats.beta.cdf
            pdf = scipy.stats.beta.pdf
            alpha, beta = a0, a1 + a2
            eps = 0.01 * alpha / (1.0 + np.sqrt(alpha))
            cdf_alpha = (cdf(x, alpha + eps, beta) - cdf(x, alpha - eps, beta)) / (2 * eps)
            cdf_x = pdf(x, alpha, beta)
            expected_grad = -cdf_alpha / cdf_x
            rel_error = np.abs(actual_grad - expected_grad) / (expected_grad + 1e-30)
            self.assertLess(np.max(rel_error), 0.001, '\n'.join([
                'Bad gradient dx[0]/dalpha[0] for Dirichlet([{}, {}, {}])'.format(a0, a1, a2),
                'x {}'.format(x),
                'expected {}'.format(expected_grad),
                'actual {}'.format(actual_grad),
                'rel error {}'.format(rel_error),
                'max error {}'.format(rel_error.max()),
                'at x={}'.format(x[rel_error.argmax()]),
            ]))

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_beta_wrt_alpha(self):
        num_samples = 20
        grid = [1e-2, 1e-1, 1e0, 1e1, 1e2]
        for con1, con0 in product(grid, grid):
            con1s = torch.tensor([con1] * num_samples, dtype=torch.float, requires_grad=True)
            con0s = con1s.new_tensor([con0] * num_samples)
            x = Beta(con1s, con0s).rsample()
            x.sum().backward()
            x, ind = x.sort()
            x = x.detach().numpy()
            actual_grad = con1s.grad[ind].numpy()
            # Compare with expected gradient dx/dcon1 along constant cdf(x,con1,con0).
            cdf = scipy.stats.beta.cdf
            pdf = scipy.stats.beta.pdf
            eps = 0.01 * con1 / (1.0 + np.sqrt(con1))
            cdf_alpha = (cdf(x, con1 + eps, con0) - cdf(x, con1 - eps, con0)) / (2 * eps)
            cdf_x = pdf(x, con1, con0)
            expected_grad = -cdf_alpha / cdf_x
            rel_error = np.abs(actual_grad - expected_grad) / (expected_grad + 1e-30)
            self.assertLess(np.max(rel_error), 0.005, '\n'.join([
                'Bad gradient dx/dcon1 for x ~ Beta({}, {})'.format(con1, con0),
                'x {}'.format(x),
                'expected {}'.format(expected_grad),
                'actual {}'.format(actual_grad),
                'rel error {}'.format(rel_error),
                'max error {}'.format(rel_error.max()),
                'at x = {}'.format(x[rel_error.argmax()]),
            ]))

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_beta_wrt_beta(self):
        num_samples = 20
        grid = [1e-2, 1e-1, 1e0, 1e1, 1e2]
        for con1, con0 in product(grid, grid):
            con0s = torch.tensor([con0] * num_samples, dtype=torch.float, requires_grad=True)
            con1s = con0s.new_tensor([con1] * num_samples)
            x = Beta(con1s, con0s).rsample()
            x.sum().backward()
            x, ind = x.sort()
            x = x.detach().numpy()
            actual_grad = con0s.grad[ind].numpy()
            # Compare with expected gradient dx/dcon0 along constant cdf(x,con1,con0).
            cdf = scipy.stats.beta.cdf
            pdf = scipy.stats.beta.pdf
            eps = 0.01 * con0 / (1.0 + np.sqrt(con0))
            cdf_beta = (cdf(x, con1, con0 + eps) - cdf(x, con1, con0 - eps)) / (2 * eps)
            cdf_x = pdf(x, con1, con0)
            expected_grad = -cdf_beta / cdf_x
            rel_error = np.abs(actual_grad - expected_grad) / (expected_grad + 1e-30)
            self.assertLess(np.max(rel_error), 0.005, '\n'.join([
                'Bad gradient dx/dcon0 for x ~ Beta({}, {})'.format(con1, con0),
                'x {}'.format(x),
                'expected {}'.format(expected_grad),
                'actual {}'.format(actual_grad),
                'rel error {}'.format(rel_error),
                'max error {}'.format(rel_error.max()),
                'at x = {!r}'.format(x[rel_error.argmax()]),
            ]))

    def test_dirichlet_multivariate(self):
        alpha_crit = 0.25 * (5.0 ** 0.5 - 1.0)
        num_samples = 100000
        for shift in [-0.1, -0.05, -0.01, 0.0, 0.01, 0.05, 0.10]:
            alpha = alpha_crit + shift
            alpha = torch.tensor([alpha], dtype=torch.float, requires_grad=True)
            alpha_vec = torch.cat([alpha, alpha, alpha.new([1])])
            z = Dirichlet(alpha_vec.expand(num_samples, 3)).rsample()
            mean_z3 = 1.0 / (2.0 * alpha + 1.0)
            loss = torch.pow(z[:, 2] - mean_z3, 2.0).mean()
            actual_grad = grad(loss, [alpha])[0]
            # Compute expected gradient by hand.
            num = 1.0 - 2.0 * alpha - 4.0 * alpha**2
            den = (1.0 + alpha)**2 * (1.0 + 2.0 * alpha)**3
            expected_grad = num / den
            self.assertEqual(actual_grad, expected_grad, atol=0.002, rtol=0, msg='\n'.join([
                "alpha = alpha_c + %.2g" % shift,
                "expected_grad: %.5g" % expected_grad,
                "actual_grad: %.5g" % actual_grad,
                "error = %.2g" % torch.abs(expected_grad - actual_grad).max(),
            ]))

    def test_dirichlet_tangent_field(self):
        num_samples = 20
        alpha_grid = [0.5, 1.0, 2.0]

        # v = dx/dalpha[0] is the reparameterized gradient aka tangent field.
        def compute_v(x, alpha):
            return torch.stack([
                _Dirichlet_backward(x, alpha, torch.eye(3, 3)[i].expand_as(x))[:, 0]
                for i in range(3)
            ], dim=-1)

        for a1, a2, a3 in product(alpha_grid, alpha_grid, alpha_grid):
            alpha = torch.tensor([a1, a2, a3], requires_grad=True).expand(num_samples, 3)
            x = Dirichlet(alpha).rsample()
            dlogp_da = grad([Dirichlet(alpha).log_prob(x.detach()).sum()],
                            [alpha], retain_graph=True)[0][:, 0]
            dlogp_dx = grad([Dirichlet(alpha.detach()).log_prob(x).sum()],
                            [x], retain_graph=True)[0]
            v = torch.stack([grad([x[:, i].sum()], [alpha], retain_graph=True)[0][:, 0]
                             for i in range(3)], dim=-1)
            # Compute ramaining properties by finite difference.
            self.assertEqual(compute_v(x, alpha), v, msg='Bug in compute_v() helper')
            # dx is an arbitrary orthonormal basis tangent to the simplex.
            dx = torch.tensor([[2., -1., -1.], [0., 1., -1.]])
            dx /= dx.norm(2, -1, True)
            eps = 1e-2 * x.min(-1, True)[0]  # avoid boundary
            dv0 = (compute_v(x + eps * dx[0], alpha) - compute_v(x - eps * dx[0], alpha)) / (2 * eps)
            dv1 = (compute_v(x + eps * dx[1], alpha) - compute_v(x - eps * dx[1], alpha)) / (2 * eps)
            div_v = (dv0 * dx[0] + dv1 * dx[1]).sum(-1)
            # This is a modification of the standard continuity equation, using the product rule to allow
            # expression in terms of log_prob rather than the less numerically stable log_prob.exp().
            error = dlogp_da + (dlogp_dx * v).sum(-1) + div_v
            self.assertLess(torch.abs(error).max(), 0.005, '\n'.join([
                'Dirichlet([{}, {}, {}]) gradient violates continuity equation:'.format(a1, a2, a3),
                'error = {}'.format(error),
            ]))


class TestDistributionShapes(DistributionsTestCase):
    def setUp(self):
        super(TestDistributionShapes, self).setUp()
        self.scalar_sample = 1
        self.tensor_sample_1 = torch.ones(3, 2)
        self.tensor_sample_2 = torch.ones(3, 2, 3)

    def tearDown(self):
        super(TestDistributionShapes, self).tearDown()

    def test_vonmises_shape_tensor_params(self):
        von_mises = VonMises(torch.tensor([0., 0.]), torch.tensor([1., 1.]))
        self.assertEqual(von_mises._batch_shape, torch.Size((2,)))
        self.assertEqual(von_mises._event_shape, torch.Size(()))
        self.assertEqual(von_mises.sample().size(), torch.Size((2,)))
        self.assertEqual(von_mises.sample(torch.Size((3, 2))).size(), torch.Size((3, 2, 2)))
        self.assertEqual(von_mises.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(von_mises.log_prob(torch.ones(2, 1)).size(), torch.Size((2, 2)))

    def test_vonmises_shape_scalar_params(self):
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

if __name__ == '__main__' and torch._C.has_lapack:
    run_tests()
