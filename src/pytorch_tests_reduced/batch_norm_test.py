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
from torch.testing._internal.common_device_type import expectedFailureXLA, dtypes, \
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
    parametrize as parametrize_test, subtest, set_default_dtype, IS_WINDOWS
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

from ..utils.pytorch_device_decorators import onlyAcceleratedDeviceTypes, instantiate_device_type_tests
from ..utils.timer_wrapper import pytorch_op_timer

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


class TestNN(NNTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    def test_batchnorm_nhwc_cpu(self):
        def helper(self, size, dtype, mixed_dtype=False):
            channels = size[1]
            input = torch.randn(size, dtype=dtype,
                                device='cpu', requires_grad=True)
            input = input.contiguous(
                memory_format=torch.channels_last).to(dtype)
            input.retain_grad()
            grad = torch.randn(size, dtype=dtype, device='cpu')
            grad = grad.contiguous(memory_format=torch.channels_last)
            bn = nn.BatchNorm2d(channels).cpu().to(dtype)
            bn.weight.data.uniform_()
            bn.bias.data.uniform_()

            ref_input = input.detach().clone().contiguous().requires_grad_(True)
            ref_grad = grad.detach().clone().contiguous()
            ref_bn = nn.BatchNorm2d(channels).cpu().to(dtype)
            ref_bn.load_state_dict(bn.state_dict())

            if mixed_dtype:
                bn.float()
                ref_bn.float()

            out = bn(input)
            out.backward(grad)
            ref_out = ref_bn(ref_input)
            ref_out.backward(ref_grad)

            self.assertTrue(out.is_contiguous(
                memory_format=torch.channels_last))
            self.assertTrue(ref_out.is_contiguous())
            self.assertEqual(out, ref_out)
            self.assertEqual(bn.weight.grad, ref_bn.weight.grad)
            self.assertEqual(bn.bias.grad, ref_bn.bias.grad)
            self.assertEqual(input.grad, ref_input.grad)

        # test NC11 and N1HW; test mixed dtype
        for shape in [(4, 8, 10, 10), (4, 1, 9, 9), (4, 9, 1, 1)]:
            helper(self, shape, torch.float, False)
            helper(self, shape, torch.bfloat16, False)
            helper(self, shape, torch.bfloat16, True)

    def test_batchnorm_non_contig_cpu(self):
        input = torch.arange(6, dtype=torch.float).reshape(1, 3, 2, 1).cpu()
        input = input.permute(0, 2, 1, 3)

        bn = torch.nn.BatchNorm2d(2).cpu().float().eval()
        bn.weight.data.uniform_()
        bn.bias.data.uniform_()

        ref_input = input.detach().clone().contiguous()
        ref_bn = nn.BatchNorm2d(2).cpu().float().eval()
        ref_bn.load_state_dict(bn.state_dict())

        out = bn(input)
        ref_out = ref_bn(ref_input)

        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(ref_out.is_contiguous())
        self.assertEqual(out, ref_out)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @unittest.skipIf(not TEST_CUDNN, "needs cudnn")
    @skipIfRocm
    def test_batchnorm_cudnn_nhwc(self):
        def run_test(input, grad_output):
            c = input.size(1)
            mod = nn.BatchNorm2d(c).cuda().float()
            mod.weight.data.uniform_()
            mod.bias.data.uniform_()
            ref_input = input.detach().clone().contiguous().requires_grad_(True)
            ref_grad = grad.detach().clone().contiguous()
            ref_mod = nn.BatchNorm2d(c).cuda().float()
            ref_mod.load_state_dict(mod.state_dict())
            out = mod(input)
            out.backward(grad_output)
            ref_out = ref_mod(ref_input)
            ref_out.backward(ref_grad)
            self.assertTrue(out.is_contiguous(
                memory_format=torch.channels_last))
            self.assertTrue(ref_out.is_contiguous())
            self.assertEqual(out, ref_out)
            self.assertEqual(mod.weight.grad, ref_mod.weight.grad)
            self.assertEqual(mod.bias.grad, ref_mod.bias.grad)
            self.assertEqual(input.grad, ref_input.grad)

        input = torch.randint(1, 10, (4, 8, 2, 2),
                              dtype=torch.float32, device="cuda")
        input = input.contiguous(
            memory_format=torch.channels_last).detach().requires_grad_()

        grad = torch.randint(1, 10, (4, 8, 2, 2),
                             dtype=torch.float32, device="cuda")
        grad = grad.contiguous(memory_format=torch.channels_last)
        run_test(input, grad)
        # see #42588, grad is channels_last contiguous, but grad.suggest_memory_format (rightly) return "contiguous"
        # not channels_last
        input = torch.randint(1, 10, (2, 8, 8, 1),
                              dtype=torch.float32, device="cuda")
        input = input.contiguous(
            memory_format=torch.channels_last).detach().requires_grad_()
        grad = torch.randint(1, 10, (2, 8, 8, 1),
                             dtype=torch.float32, device="cuda")
        grad = grad.permute(0, 2, 1, 3)
        run_test(input, grad)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_batchnorm_cudnn_half(self):
        # THNN
        input = torch.randint(
            1, 10, (2, 3, 2, 2), dtype=torch.half, device="cuda", requires_grad=True)
        m = nn.BatchNorm2d(3).half().cuda()
        thnn_output = m(input)
        thnn_output.sum().backward()
        thnn_input_grad = input.grad.data.clone()
        self.assertEqualTypeString(thnn_output, input)
        # cuDNN
        if TEST_CUDNN:
            input.grad = None
            m = m.float()
            cudnn_output = m(input)
            cudnn_output.sum().backward()
            cudnn_input_grad = input.grad.data.clone()
            self.assertEqualTypeString(cudnn_output, input)
            self.assertEqual(cudnn_output, thnn_output)
            self.assertEqual(cudnn_input_grad,
                             thnn_input_grad, atol=1e-3, rtol=0)

    @onlyAcceleratedDeviceTypes
    def test_batchnorm_nonaffine_cuda_half_input(self, device):
        input = torch.randn(16, 3, 24, 24, dtype=torch.half, device=device)
        # keep running stats in FP32
        m = nn.BatchNorm2d(3, affine=False).to(device).float()
        output = m(input)
        self.assertEqualTypeString(output, input)
        m.eval()
        output = m(input)
        self.assertEqualTypeString(output, input)

    def test_batchnorm_raises_error_if_less_than_one_value_per_channel(self, device):
        x = torch.rand(10, device=device)[None, :, None]
        with self.assertRaises(ValueError):
            torch.nn.BatchNorm1d(10)(x)

    def test_batchnorm_raises_error_if_running_mean_is_not_same_size_as_input(self, device):
        input = torch.rand(2, 10, device=device)
        running_var = torch.rand(10, device=device)
        wrong_sizes = [9, 11]
        for size in wrong_sizes:
            with self.assertRaises(RuntimeError):
                F.batch_norm(input, torch.rand(size), running_var)

    def test_batchnorm_raises_error_if_running_var_is_not_same_size_as_input(self, device):
        input = torch.rand(2, 10, device=device)
        running_mean = torch.rand(10, device=device)
        wrong_sizes = [9, 11]
        for size in wrong_sizes:
            with self.assertRaises(RuntimeError):
                F.batch_norm(input, running_mean, torch.rand(size))

    def test_batchnorm_raises_error_if_weight_is_not_same_size_as_input(self, device):
        input = torch.rand(2, 10, device=device)
        running_mean = torch.rand(10, device=device)
        running_var = torch.rand(10, device=device)
        wrong_sizes = [9, 11]
        for size in wrong_sizes:
            with self.assertRaises(RuntimeError):
                F.batch_norm(input, running_mean, running_var,
                             weight=Parameter(torch.rand(size)))

    def test_batchnorm_raises_error_if_bias_is_not_same_size_as_input(self, device):
        input = torch.rand(2, 10, device=device)
        running_mean = torch.rand(10, device=device)
        running_var = torch.rand(10, device=device)
        wrong_sizes = [9, 11]
        for size in wrong_sizes:
            with self.assertRaises(RuntimeError):
                F.batch_norm(input, running_mean, running_var,
                             bias=Parameter(torch.rand(size)))

    def test_batchnorm_raises_error_if_running_var_or_running_mean_have_forward_grad(self, device):
        args = (
            torch.randn(3, 2, 5, device=device),  # input
            torch.randn(2, device=device),  # running_mean
            torch.randn(2, device=device),  # running_var
        )
        kwargs = {'training': False, 'momentum': -1.2}
        fn = partial(F.batch_norm, **kwargs)

        for dual_indices in ((0,), (1,), (1, 2), (0, 1), (0, 1, 2),):
            tangents = tuple(torch.rand_like(x) for x in args)

            with fwAD.dual_level():
                duals = [fwAD.make_dual(primal, tangent) if i in dual_indices else primal
                         for i, (primal, tangent) in enumerate(zip(args, tangents))]
                msg = "batch_norm is not differentiable wrt running_mean and running_var"
                # 0 needs to have forward grad because otherwise we won't even run batch_norm_jvp
                if (1 in dual_indices or 2 in dual_indices) and 0 in dual_indices:
                    with self.assertRaisesRegex(RuntimeError, msg):
                        fn(*duals)
                else:
                    fn(*duals)

    def test_batchnorm_buffer_update_when_stats_are_not_tracked(self, device):
        input_size = (32, 4)
        # Instantiate BN with buffers that are not None
        bn = nn.BatchNorm1d(input_size[1], track_running_stats=True)
        # Use buffers for normalization but don't update them
        bn.track_running_stats = False
        # Store initial values
        num_batches = bn.num_batches_tracked.clone()
        running_mean = bn.running_mean.clone()
        running_var = bn.running_var.clone()
        # Forward random tensor
        _ = bn(torch.rand(input_size, device=device))
        # Ensure none of the buffers has been updated
        self.assertTrue(torch.equal(num_batches, bn.num_batches_tracked))
        self.assertTrue(torch.equal(running_mean, bn.running_mean))
        self.assertTrue(torch.equal(running_var, bn.running_var))

    @onlyAcceleratedDeviceTypes
    def test_batchnorm_nhwc_cuda(self, device):
        for dtype in (torch.half, torch.float):
            (N, C, H, W) = 2, 64, 50, 50
            model = torch.nn.BatchNorm2d(
                C, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            model = model.eval().to(device).to(dtype)
            inp1 = torch.randn(
                N, C, H, W, device=device, dtype=dtype)
            inp2 = inp1.contiguous(memory_format=torch.channels_last)
            out1 = model(inp1)
            out2 = model(inp2)
            self.assertTrue(torch.equal(out1, out2))


instantiate_device_type_tests(TestNN, globals())

if __name__ == '__main__':
    run_tests()
