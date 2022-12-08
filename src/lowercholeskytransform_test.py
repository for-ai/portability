# Owner(s): ["module: distributions"]

import io
from numbers import Number

import pytest

import torch
from torch.autograd.functional import jacobian
from torch.distributions import Dirichlet, Independent, Normal, TransformedDistribution, constraints
from torch.distributions.transforms import (AbsTransform, AffineTransform, ComposeTransform,
                                            CorrCholeskyTransform, CumulativeDistributionTransform,
                                            ExpTransform, IndependentTransform,
                                            LowerCholeskyTransform, PowerTransform,
                                            ReshapeTransform, SigmoidTransform, TanhTransform,
                                            SoftmaxTransform, SoftplusTransform, StickBreakingTransform,
                                            identity_transform, Transform, _InverseTransform)
from torch.distributions.utils import tril_matrix_to_vec, vec_to_tril_matrix



def test_compose_transform_shapes():
    transform0 = ExpTransform()
    transform1 = SoftmaxTransform()
    transform2 = LowerCholeskyTransform()

    assert transform0.event_dim == 0
    assert transform1.event_dim == 1
    assert transform2.event_dim == 2
    assert ComposeTransform([transform0, transform1]).event_dim == 1
    assert ComposeTransform([transform0, transform2]).event_dim == 2
    assert ComposeTransform([transform1, transform2]).event_dim == 2


transform0 = ExpTransform()
transform1 = SoftmaxTransform()
transform2 = LowerCholeskyTransform()
base_dist0 = Normal(torch.zeros(4, 4), torch.ones(4, 4))
base_dist1 = Dirichlet(torch.ones(4, 4))
base_dist2 = Normal(torch.zeros(3, 4, 4), torch.ones(3, 4, 4))


@pytest.mark.parametrize('batch_shape, event_shape, dist', [
    ((4, 4), (), base_dist0),
    ((4,), (4,), base_dist1),
    ((4, 4), (), TransformedDistribution(base_dist0, [transform0])),
    ((4,), (4,), TransformedDistribution(base_dist0, [transform1])),
    ((4,), (4,), TransformedDistribution(base_dist0, [transform0, transform1])),
    ((), (4, 4), TransformedDistribution(base_dist0, [transform0, transform2])),
    ((4,), (4,), TransformedDistribution(base_dist0, [transform1, transform0])),
    ((), (4, 4), TransformedDistribution(base_dist0, [transform1, transform2])),
    ((), (4, 4), TransformedDistribution(base_dist0, [transform2, transform0])),
    ((), (4, 4), TransformedDistribution(base_dist0, [transform2, transform1])),
    ((4,), (4,), TransformedDistribution(base_dist1, [transform0])),
    ((4,), (4,), TransformedDistribution(base_dist1, [transform1])),
    ((), (4, 4), TransformedDistribution(base_dist1, [transform2])),
    ((4,), (4,), TransformedDistribution(base_dist1, [transform0, transform1])),
    ((), (4, 4), TransformedDistribution(base_dist1, [transform0, transform2])),
    ((4,), (4,), TransformedDistribution(base_dist1, [transform1, transform0])),
    ((), (4, 4), TransformedDistribution(base_dist1, [transform1, transform2])),
    ((), (4, 4), TransformedDistribution(base_dist1, [transform2, transform0])),
    ((), (4, 4), TransformedDistribution(base_dist1, [transform2, transform1])),
    ((3, 4, 4), (), base_dist2),
    ((3,), (4, 4), TransformedDistribution(base_dist2, [transform2])),
    ((3,), (4, 4), TransformedDistribution(base_dist2, [transform0, transform2])),
    ((3,), (4, 4), TransformedDistribution(base_dist2, [transform1, transform2])),
    ((3,), (4, 4), TransformedDistribution(base_dist2, [transform2, transform0])),
    ((3,), (4, 4), TransformedDistribution(base_dist2, [transform2, transform1])),
])
def test_transformed_distribution_shapes(batch_shape, event_shape, dist):
    assert dist.batch_shape == batch_shape
    assert dist.event_shape == event_shape
    x = dist.rsample()
    try:
        dist.log_prob(x)  # this should not crash
    except NotImplementedError:
        pytest.skip('Not implemented.')


if __name__ == '__main__':
    pytest.main([__file__])
