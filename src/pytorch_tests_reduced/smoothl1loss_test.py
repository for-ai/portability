# Owner(s): ["module: nn"]

from torch.types import _TensorOrTensors
from torch.testing._internal.common_cuda import tf32_on_and_off, tf32_is_not_fp32, tf32_off, tf32_on
from torch.testing._internal.common_utils import dtype2prec_DONTUSE
from torch.testing._internal.common_utils import _assertGradAndGradgradChecks, gradcheck, gradgradcheck, \
    GRADCHECK_NONDET_TOL
import torch.testing._internal.hypothesis_utils as hu
from hypothesis import given
from torch.nn import MultiheadAttention
from torch.testing._internal.common_device_type import dtypes, \
    dtypesIfCUDA, precisionOverride, skipCUDAIfCudnnVersionLessThan, onlyCUDA, onlyCPU, \
    skipCUDAIfRocm, skipCUDAIf, skipCUDAIfNotRocm, \
    onlyNativeDeviceTypes, deviceCountAtLeast, largeTensorTest, expectedFailureMeta, skipMeta, get_all_device_types
from .common_nn import NNTestCase, NewModuleTest, CriterionTest, \
    module_tests, criterion_tests, loss_reference_fns, \
    ctcloss_reference, new_module_tests, single_batch_reference_fn, _test_bfloat16_ops, _test_module_empty_input
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU, TEST_CUDNN, TEST_CUDNN_VERSION
from torch.testing._internal.common_utils import freeze_rng_state, run_tests, TestCase, skipIfNoLapack, skipIfRocm, \
    TEST_NUMPY, TEST_SCIPY, TEST_WITH_CROSSREF, TEST_WITH_ROCM, \
    download_file, get_function_arglist, load_tests, skipIfMps,\
    TemporaryFileName, TEST_WITH_UBSAN, IS_PPC, \
    parametrize as parametrize_test, subtest, instantiate_parametrized_tests, IS_WINDOWS
from torch.testing._internal.common_dtype import integral_types, get_all_math_dtypes
from torch.nn.parallel._functions import Broadcast
from torch.nn import Parameter
from torch.nn.utils.fusion import fuse_linear_bn_weights
from torch.nn.utils.fusion import fuse_conv_bn_weights
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
from itertools import product
from functools import reduce, partial
from operator import mul
from collections import OrderedDict
from tempfile import NamedTemporaryFile
import weakref
import gc

import torch
from ..utils.pytorch_device_decorators import onlyNativeDeviceTypes, onlyAcceleratedDeviceTypes, instantiate_device_type_tests
from ..utils.timer_wrapper import pytorch_op_timer

# TODO: remove this global setting
# NN tests use double as the default dtype


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
    def test_nn_scalars_reductions(self, device):
        torch.set_default_dtype(torch.double)
        # One off tests to ensure scalars from nn.yaml are properly applied
        def verify_reduction_scalars(input, reduction, output):
            if reduction != 'none' or input.dim() == 0:
                self.assertEqual((), output.shape)
            else:
                self.assertNotEqual((), output.shape)
            output.sum().backward()
            self.assertEqual(input.shape, input.grad.shape)

        for input_shape in [(5, 6), ()]:
            for reduction in ['none', 'mean', 'sum']:
                for module in [torch.nn.SmoothL1Loss]:
                    input = torch.randn(
                        input_shape, device=device, requires_grad=True)
                    target = torch.empty(input_shape, device=device).random_(2)
                    sigmoid = nn.Sigmoid()

                    input = torch.randn(
                        input_shape, device=device, requires_grad=True)
                    with pytorch_op_timer():
                        m = module(reduction=reduction)
                    output = m(sigmoid(input), target)
                    verify_reduction_scalars(input, reduction, output)
        torch.set_default_dtype(torch.float)

    @onlyNativeDeviceTypes
    def test_smooth_l1_loss_vs_huber_loss(self, device):
        torch.set_default_dtype(torch.double)
        def _make_test_tensor(shape, contiguous=True):
            if contiguous:
                test_tensor = torch.randn(shape, device=device)
            else:
                # Select every other element in the innermost dimension to
                # make it non-contiguous.
                doubled_shape = list(shape)
                doubled_shape[-1] *= 2
                test_tensor = torch.randn(doubled_shape, device=device)
                test_tensor = test_tensor[..., ::2]
            return test_tensor

        def _test_smooth_l1_loss_vs_huber_loss_helper(input, target, beta, require_equal):
            for reduction in ['mean', 'sum', 'none']:
                with pytorch_op_timer():
                    smooth_l1 = torch.nn.SmoothL1Loss(
                    beta=beta, reduction=reduction)
                # beta hyper-parameter is called delta for Huber
                huber = torch.nn.HuberLoss(delta=beta, reduction=reduction)
                smooth_l1_loss = smooth_l1(input, target)
                huber_loss = huber(input, target)

                if require_equal:

                    self.assertEqual(smooth_l1_loss, huber_loss)
                else:
                    # Huber loss should be larger than smooth L1 loss by a factor of beta.
                    self.assertEqual(smooth_l1_loss * beta, huber_loss)

        def _test_smooth_l1_loss_vs_huber_loss_multi_input_helper(beta, require_equal):
            # Test the non-vectorized case.
            shape = (2, 2)
            _test_smooth_l1_loss_vs_huber_loss_helper(input=_make_test_tensor(shape),
                                                      target=_make_test_tensor(
                                                          shape),
                                                      beta=beta,
                                                      require_equal=require_equal)

            # Test the vectorized case (innermost dim > 32).
            shape = (64, 64)
            _test_smooth_l1_loss_vs_huber_loss_helper(input=_make_test_tensor(shape),
                                                      target=_make_test_tensor(
                                                          shape),
                                                      beta=beta,
                                                      require_equal=require_equal)

            # Test the non-contiguous case.
            _test_smooth_l1_loss_vs_huber_loss_helper(input=_make_test_tensor(shape, contiguous=False),
                                                      target=_make_test_tensor(
                                                          shape, contiguous=False),
                                                      beta=beta,
                                                      require_equal=require_equal)

        def test_equal_when_beta_is_one():
            _test_smooth_l1_loss_vs_huber_loss_multi_input_helper(
                beta=1.0, require_equal=True)

        def test_unequal_when_beta_is_less_than_one():
            _test_smooth_l1_loss_vs_huber_loss_multi_input_helper(
                beta=0.5, require_equal=False)

        def test_unequal_when_beta_is_greater_than_one():
            _test_smooth_l1_loss_vs_huber_loss_multi_input_helper(
                beta=1.5, require_equal=False)

        test_equal_when_beta_is_one()
        test_unequal_when_beta_is_less_than_one()
        test_unequal_when_beta_is_greater_than_one()
        torch.set_default_dtype(torch.float)

    # @onlyCPU
    def test_smooth_l1_loss_bfloat16(self, device):
        torch.set_default_dtype(torch.double)
        def test_dtype(fn, input, target, dtype):
            input = input.detach().clone().to(dtype=dtype).requires_grad_(True)
            input2 = input.detach().clone().float().requires_grad_(True)
            target = target.detach().clone().to(dtype=dtype)
            target2 = target.detach().clone().float()
            out = fn(input, target)
            out.sum().backward()
            out2 = fn(input2, target2)
            out2.sum().backward()
            self.assertEqual(out.dtype, dtype)
            self.assertEqual(input.grad.dtype, dtype)
            self.assertEqual(out, out2, exact_dtype=False)
            self.assertEqual(input.grad, input2.grad, exact_dtype=False)

        def func(device):
            with pytorch_op_timer():
                return nn.SmoothL1Loss().to(device=device)

        shapes = [[1, 3, 1, 6], [1, 3, 1, 128], [1, 3, 128, 128]]
        for shape in shapes:
            x = torch.randn(shape, device=device, requires_grad=True)
            t = torch.randn(shape, device=device)
            test_dtype(func(device), x, t, torch.bfloat16)
        torch.set_default_dtype(torch.float)


instantiate_device_type_tests(TestNNDeviceType, globals())

if __name__ == '__main__':
    run_tests()
