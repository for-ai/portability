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


class TestNNDeviceType(NNTestCase):
    def test_invalid_conv3d(self):
        for dtype in [torch.bfloat16, torch.float, torch.double, torch.cfloat, torch.cdouble]:
            module = torch.nn.Conv3d(
                1, 1, kernel_size=3, dilation=2, stride=2).to(dtype)
            input = torch.empty(1, 1, 4, 4, 4).to(dtype)
            self.assertRaises(RuntimeError, lambda: module(input))

            # Negative stride check
            module = torch.nn.Conv3d(1, 1, kernel_size=3, stride=-2)
            input = torch.empty(1, 1, 4, 4, 4)
            with self.assertRaisesRegex(RuntimeError, 'non-positive stride is not supported'):
                module(input)

    @dtypes(torch.float, torch.cfloat)
    def test_conv3d_same_padding(self, device, dtype):
        if dtype is torch.cfloat:
            rtol, atol = 2e-6, 2e-6
        else:
            rtol, atol = None, None
        # Compare F.conv3d padding='same' output against manual padding
        # Without strides/dilation
        x = torch.rand(1, 1, 10, 11, 12, device=device, dtype=dtype)
        y = torch.rand(1, 1, 1, 2, 5, device=device, dtype=dtype)
        expect = F.conv3d(x, y, padding=(0, 1, 2))[..., :, 1:, :]
        actual = F.conv3d(x, y, padding='same')
        self.assertEqual(expect, actual, rtol=rtol, atol=atol)

        # With dilation
        expect = F.conv3d(x, y, padding=(0, 1, 4), dilation=2)
        actual = F.conv3d(x, y, padding='same', dilation=2)
        self.assertEqual(expect, actual, rtol=rtol, atol=atol)

        # Dilation with asymmetric padding
        y = torch.rand(1, 1, 4, 4, 4, device=device, dtype=dtype)
        expect = F.conv3d(x, y, padding=5, dilation=3)[..., 1:, 1:, 1:]
        actual = F.conv3d(x, y, padding='same', dilation=3)
        self.assertEqual(expect, actual, rtol=rtol, atol=atol)

    @dtypes(torch.float, torch.cfloat)
    def test_conv3d_valid_padding(self, device, dtype):
        # Test F.conv3d padding='valid' is the same as no padding
        x = torch.rand(1, 1, 1, 1, 10, dtype=dtype, device=device)
        y = torch.rand(1, 1, 1, 1, 4, dtype=dtype, device=device)
        expect = F.conv3d(x, y)
        actual = F.conv3d(x, y, padding='valid')
        self.assertEqual(expect, actual)

    @dtypes(torch.double, torch.cdouble)
    def test_conv3d_same_padding_backward(self, device, dtype):
        check_forward_ad = torch.device(device).type != 'xla'

        # Test F.conv3d gradients work with padding='same'
        x = torch.rand(1, 1, 1, 11, 12, dtype=dtype,
                       device=device, requires_grad=True)
        y = torch.rand(1, 1, 1, 2, 5, dtype=dtype,
                       device=device, requires_grad=True)

        # Symmetric padding
        z = F.conv3d(x, y, padding=(0, 1, 4), dilation=2)
        z.sum().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        z = F.conv3d(x, y, padding='same', dilation=2)
        z.sum().backward()
        self.assertEqual(gx_expect, x.grad)
        self.assertEqual(gy_expect, y.grad)
        x.grad, y.grad = None, None

        gradcheck(lambda x, y: F.conv3d(x, y, padding='same', dilation=2), (x, y),
                  check_forward_ad=check_forward_ad, nondet_tol=1e-5)
        if torch.device(device).type != 'cuda':
            # https://github.com/pytorch/pytorch/issues/70702
            gradgradcheck(lambda x, y: F.conv3d(x, y, padding='same', dilation=2), (x, y),
                          check_fwd_over_rev=True)

        # Asymmetric padding
        y = torch.rand(1, 1, 1, 4, 4, dtype=dtype,
                       device=device, requires_grad=True)
        z = F.conv3d(x, y, padding=2)[..., 1:, 1:]
        z.sum().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        z = F.conv3d(x, y, padding='same')
        z.sum().backward()
        self.assertEqual(gx_expect, x.grad)
        self.assertEqual(gy_expect, y.grad)

        gradcheck(lambda x, y: F.conv3d(x, y, padding='same'), (x, y),
                  check_forward_ad=check_forward_ad, nondet_tol=1e-5)
        if torch.device(device).type != 'cuda':
            # https://github.com/pytorch/pytorch/issues/70702
            gradgradcheck(lambda x, y: F.conv3d(x, y, padding='same'), (x, y),
                          check_fwd_over_rev=True)

    @unittest.skipIf(not TEST_SCIPY, "Scipy required for the test.")
    @dtypes(torch.float, torch.cfloat)
    @parametrize_test("mode", ('valid', 'same'))
    def test_conv3d_vs_scipy(self, device, dtype, mode):
        t = make_tensor((1, 5, 5, 10), device=device, dtype=dtype)
        weight_even = make_tensor((1, 1, 2, 2, 4), device=device, dtype=dtype)
        weight_odd = make_tensor((1, 1, 2, 3, 5), device=device, dtype=dtype)

        def _test(t, weight, mode):
            # SciPy expects two 3-D inputs.
            t_a = t.squeeze(0).cpu().numpy()
            w_a = weight.squeeze(0).squeeze(0).cpu().numpy()
            expected = scipy.signal.convolve(t_a, w_a, mode=mode)

            kwargs = {'padding': mode}
            if mode == 'same':
                # `same` padding in PyTorch conv3d is different
                # from SciPy
                left_right_pad = weight.shape[4] // 2
                top_bottom_pad = weight.shape[3] // 2
                front_back_pad = weight.shape[2] // 2
                p = (left_right_pad, left_right_pad, top_bottom_pad, top_bottom_pad,
                     front_back_pad, front_back_pad)
                t = torch.nn.functional.pad(t, p)
                # We have already taken care of padding
                kwargs.pop("padding")

            # second input is flipped in SciPy's convolve
            weight_flipped = torch.flip(weight, (2, 3, 4))
            actual = torch.nn.functional.conv3d(
                t, weight_flipped, **kwargs).squeeze(0)
            if mode == 'same':
                actual = actual[:5, :5, :10]

            self.assertEqual(actual, expected, rtol=2e-5, atol=5e-6)

        # Global dtype for this test suite is torch.double
        # This leads to change in type-promotion
        # and conv1d outputs `complex128` for `complex64` input.
        with set_default_dtype(torch.float):
            _test(t, weight_even, mode)
            _test(t, weight_odd, mode)

    @dtypes(torch.double, torch.cdouble)
    def test_conv3d_valid_padding_backward(self, device, dtype):
        check_forward_ad = torch.device(device).type != 'xla'

        # Test F.conv3d gradients work with padding='valid'
        x = torch.rand(1, 1, 1, 1, 10, dtype=dtype,
                       device=device, requires_grad=True)
        y = torch.rand(1, 1, 1, 1, 4, dtype=dtype,
                       device=device, requires_grad=True)
        F.conv3d(x, y, padding=0).sum().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        F.conv3d(x, y, padding='valid').sum().backward()
        gx_actual, gy_actual = x.grad, y.grad
        self.assertEqual(gx_expect, gx_actual)
        self.assertEqual(gy_expect, gy_actual)

        gradcheck(lambda x, y: F.conv3d(x, y, padding='valid'),
                  (x, y), check_forward_ad=check_forward_ad)
        gradgradcheck(lambda x, y: F.conv3d(x, y, padding='valid'),
                      (x, y), check_fwd_over_rev=check_forward_ad)


instantiate_device_type_tests(TestNNDeviceType, globals())

if __name__ == '__main__':
    run_tests()
