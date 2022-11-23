# Owner(s): ["module: nn"]

from torch.types import _TensorOrTensors
from torch.testing._internal.common_cuda import tf32_on_and_off, tf32_is_not_fp32, tf32_off, tf32_on
from torch.testing._internal.common_utils import dtype2prec_DONTUSE
from torch.testing._internal.common_utils import _assertGradAndGradgradChecks, gradcheck, gradgradcheck, \
    GRADCHECK_NONDET_TOL
import torch.testing._internal.hypothesis_utils as hu
from torch.testing import make_tensor
from hypothesis import given
from torch.nn import MultiheadAttention
from torch.testing._internal.common_device_type import expectedFailureXLA, instantiate_device_type_tests, dtypes, \
    dtypesIfCUDA, precisionOverride, skipCUDAIfNoCudnn, skipCUDAIfCudnnVersionLessThan, onlyCUDA, onlyCPU, \
    skipCUDAIfRocm, skipCUDAIf, skipCUDAIfNotRocm, skipCUDAIfRocmVersionLessThan, skipCUDAIfNotMiopenSuggestNHWC, \
    onlyNativeDeviceTypes, deviceCountAtLeast, largeTensorTest, expectedFailureMeta, skipMeta, get_all_device_types, \
    disableMkldnn, skipCPUIfNoMkldnn, disablecuDNN, skipCUDAIfMiopen, skipCUDAIfNoMiopen
from torch.testing._internal.common_nn import NNTestCase, NewModuleTest, CriterionTest, \
    module_tests, criterion_tests, loss_reference_fns, \
    ctcloss_reference, new_module_tests, single_batch_reference_fn
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU, TEST_CUDNN, TEST_CUDNN_VERSION
from torch.testing._internal.common_utils import freeze_rng_state, run_tests, TestCase, skipIfNoLapack, skipIfRocm, \
    skipIfRocmVersionLessThan, skipIfNotMiopenSuggestNHWC, TEST_NUMPY, TEST_SCIPY, TEST_WITH_CROSSREF, TEST_WITH_ROCM, \
    download_file, get_function_arglist, load_tests, skipIfMps,\
    suppress_warnings, TemporaryFileName, TEST_WITH_UBSAN, IS_PPC, \
    parametrize as parametrize_test, subtest, instantiate_parametrized_tests, set_default_dtype, IS_WINDOWS
from torch.testing._internal.common_dtype import integral_types, floating_types_and, get_all_math_dtypes, \
    floating_and_complex_types_and
from torch.nn.parallel._functions import Broadcast
from torch.nn.parameter import UninitializedParameter, UninitializedBuffer
from torch.nn import Parameter
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn.utils.prune as prune
import torch.nn.utils.parametrize as parametrize
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import torch.nn.utils.rnn as rnn_utils
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.autograd.forward_ad as fwAD
from torch._six import inf, nan
import contextlib
import math
import random
import string
import unittest
import io
import unittest.mock as mock
import itertools
import warnings
import pickle
from copy import deepcopy
from itertools import repeat, product
from functools import reduce, partial
from operator import mul
from collections import OrderedDict
from tempfile import NamedTemporaryFile

import torch

# TODO: remove this global setting
# NN tests use double as the default dtype
torch.set_default_dtype(torch.double)


AMPERE_OR_ROCM = TEST_WITH_ROCM or tf32_is_not_fp32()

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

if TEST_SCIPY:
    from scipy import stats
    import scipy.signal
    import scipy.ndimage

if TEST_NUMPY:
    import numpy as np


# WARNING: If you add a new top-level test case to this file, you MUST
# update test/run_test.py to list it, otherwise it will NOT be run in
# CI.


class TestNNDeviceType(NNTestCase):
    def _test_LayerNorm_general(self, device, dtype=torch.float):
        for i in range(2, 6):
            shape = torch.randint(3, 6, (i,), dtype=torch.long).tolist()
            x = torch.empty(*shape, device=device, dtype=dtype).uniform_(0, 10)
            normalized_ndim = random.randint(1, i - 1)  # inclusive
            normalized_shape = shape[-normalized_ndim:]
            unnormalized_shape = shape[:-normalized_ndim]

            # test that LN normalizes to mean 0 and stddev 1
            ln = nn.LayerNorm(normalized_shape, eps=0).to(device, dtype)
            ln.weight.data.fill_(1)
            ln.bias.data.fill_(0)
            output = ln(x)
            out_reshaped = output.view(*(unnormalized_shape + [-1]))
            mean = out_reshaped.mean(-1)
            var = out_reshaped.var(-1, unbiased=False)

            delta = 1e-1 if dtype == torch.bfloat16 else 1e-5
            self.assertEqual(torch.abs(mean.data).mean(),
                             0, atol=delta, rtol=0)
            self.assertEqual(torch.abs(var.data).mean(), 1, atol=delta, rtol=0)

            # test that LN applies weight and bias correctly
            scale, bias = torch.empty(2).uniform_(0.2, 2).tolist()
            ln.weight.data.fill_(scale)
            ln.bias.data.fill_(bias)
            output = ln(x)
            out_reshaped = output.view(*(unnormalized_shape + [-1]))
            mean = out_reshaped.mean(-1)
            var = out_reshaped.var(-1, unbiased=False)
            self.assertEqual(torch.abs(mean.data).mean(),
                             bias, atol=delta, rtol=0)
            self.assertEqual(torch.abs(var.data).mean(),
                             scale ** 2, atol=delta, rtol=0)

        bad_norm_shape_input_shape = {
            (): (),
            (2, 3): (3,),
            (2,): (1, 2, 3),
            (10,): (2, 3),
            10: (2, 3),
        }
        for norm_shape, input_shape in bad_norm_shape_input_shape.items():
            ln = nn.LayerNorm(norm_shape)
            input = torch.empty(input_shape, device=device,
                                dtype=dtype).uniform_(0, 10)
            self.assertRaises(RuntimeError, lambda: ln(input))

    def _test_LayerNorm_cuda_half(self, device):
        input = torch.empty(2, 3, 3, 2, device=device, dtype=torch.half).random_(
            1, 10).requires_grad_(True)
        m = nn.LayerNorm([3, 2]).to(device, torch.half)
        output = m(input)
        output.sum().backward()
        self.assertEqualTypeString(output, input)

    def _test_LayerNorm_cpu_mixed_dtype(self, device):
        for elementwise_affine in [True, False]:
            # layer norm input shape is normalized to m x n, cpu vectorized on n,
            # so make sure n exceeds vector length
            input = torch.empty(2, 3, 11, 3, device=device,
                                dtype=torch.bfloat16).random_(1, 10)
            m = nn.LayerNorm([11, 3], elementwise_affine=elementwise_affine).to(
                device, torch.bfloat16)
            m2 = deepcopy(m).to(device, torch.float)
            out = m(input)
            out2 = m2(input)
            self.assertEqual(out, out2)

    @contextlib.contextmanager
    def torch_vital_set(value):
        stash = None
        if 'TORCH_VITAL' in os.environ:
            stash = os.environ['TORCH_VITAL']
        os.environ['TORCH_VITAL'] = value
        try:
            yield
        finally:
            if stash:
                os.environ['TORCH_VITAL'] = stash
            else:
                del os.environ['TORCH_VITAL']

    def test_LayerNorm_general(self, device):
        self._test_LayerNorm_general(device)

        if self.device_type == 'cuda' or self.device_type == 'cpu':
            self._test_LayerNorm_general(device, dtype=torch.bfloat16)

        if self.device_type == 'cuda':
            self._test_LayerNorm_cuda_half(device)

        if self.device_type == 'cpu':
            self._test_LayerNorm_cpu_mixed_dtype(device)

    @onlyNativeDeviceTypes
    def test_LayerNorm_numeric(self, device):
        def layer_norm_ref(X, gamma, beta, normalized_shape, eps):
            feature_size = np.prod(normalized_shape)
            X_view = X.view(-1, feature_size)
            mean = X_view.mean(dim=-1, keepdim=True)
            var = X_view.var(dim=-1, unbiased=False, keepdim=True)
            Y = (X_view - mean) / torch.sqrt(var + eps)
            Y = Y * gamma.view(-1) + beta.view(-1)
            return Y.view(*X.size())

        normalized_shape = [256, 256, 144]
        layer_norm = nn.LayerNorm(normalized_shape).float().to(device)
        X = torch.rand(2, *normalized_shape, dtype=torch.float32,
                       device=device)

        Y = layer_norm(X)
        Y_ref = layer_norm_ref(X, layer_norm.weight.data, layer_norm.bias.data,
                               normalized_shape, layer_norm.eps)
        self.assertEqual(Y, Y_ref, rtol=0, atol=1e-5)

        if self.device_type == 'cuda':
            layer_norm.cpu()
            Y_cpu = layer_norm(X.cpu())
            self.assertEqual(Y_cpu, Y, rtol=0, atol=1e-5)


instantiate_device_type_tests(TestNNDeviceType, globals())
if __name__ == '__main__':
    run_tests()
