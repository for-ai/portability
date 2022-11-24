# Owner(s): ["module: nn"]

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

# TODO: remove this global setting
# NN tests use double as the default dtype
torch.set_default_dtype(torch.double)

from torch._six import inf, nan
import torch.autograd.forward_ad as fwAD
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import torch.nn.utils.parametrize as parametrize
import torch.nn.utils.prune as prune
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.nn.utils.fusion import fuse_conv_bn_weights
from torch.nn.utils.fusion import fuse_linear_bn_weights
from torch.nn import Parameter
from torch.nn.parallel._functions import Broadcast
from torch.testing._internal.common_dtype import integral_types, get_all_math_dtypes
from torch.testing._internal.common_utils import freeze_rng_state, run_tests, TestCase, skipIfNoLapack, skipIfRocm, \
    TEST_NUMPY, TEST_SCIPY, TEST_WITH_CROSSREF, TEST_WITH_ROCM, \
    download_file, get_function_arglist, load_tests, skipIfMps,\
    TemporaryFileName, TEST_WITH_UBSAN, IS_PPC, \
    parametrize as parametrize_test, subtest, instantiate_parametrized_tests, IS_WINDOWS
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU, TEST_CUDNN, TEST_CUDNN_VERSION
from torch.testing._internal.common_nn import NNTestCase, NewModuleTest, CriterionTest, \
    module_tests, criterion_tests, loss_reference_fns, \
    ctcloss_reference, new_module_tests, single_batch_reference_fn
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes, \
    dtypesIfCUDA, precisionOverride, skipCUDAIfCudnnVersionLessThan, onlyCUDA, onlyCPU, \
    skipCUDAIfRocm, skipCUDAIf, skipCUDAIfNotRocm, \
    onlyNativeDeviceTypes, deviceCountAtLeast, largeTensorTest, expectedFailureMeta, skipMeta, get_all_device_types
from torch.nn import MultiheadAttention

from torch.testing._internal.common_utils import _assertGradAndGradgradChecks, gradcheck, gradgradcheck, \
    GRADCHECK_NONDET_TOL
from torch.testing._internal.common_utils import dtype2prec_DONTUSE
from torch.testing._internal.common_cuda import tf32_on_and_off, tf32_is_not_fp32, tf32_off, tf32_on
from torch.types import _TensorOrTensors


AMPERE_OR_ROCM = TEST_WITH_ROCM or tf32_is_not_fp32()

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

class TestNN(NNTestCase):
   

    def test_hook_cpp(self):
        bn = nn.BatchNorm1d(5)

        def hook(module, grad_inputs, grad_outputs):
            self.assertEqual(len(grad_inputs), 1)
            self.assertEqual(len(grad_outputs), 1)
            self.assertEqual(module, bn)

        bn.register_full_backward_hook(hook)
        output = bn(torch.randn(5, 5, requires_grad=True))
        output.sum().backward()


    def test_batchnorm_raises_error_if_less_than_one_value_per_channel(self):
        x = torch.rand(10)[None, :, None]
        with self.assertRaises(ValueError):
            torch.nn.BatchNorm1d(10)(x)

    def test_batchnorm_buffer_update_when_stats_are_not_tracked(self):
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
        _ = bn(torch.rand(input_size))
        # Ensure none of the buffers has been updated
        self.assertTrue(torch.equal(num_batches, bn.num_batches_tracked))
        self.assertTrue(torch.equal(running_mean, bn.running_mean))
        self.assertTrue(torch.equal(running_var, bn.running_var))

    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_convert_sync_batchnorm(self):
        module = torch.nn.Sequential(
            torch.nn.BatchNorm1d(100),
            torch.nn.InstanceNorm1d(100)
        ).cuda()

        # necessary to have an anchor point for comparison, in case the
        # convert_sync_batchnorm updates in place
        comp_module = torch.nn.Sequential(
            torch.nn.BatchNorm1d(100),
            torch.nn.InstanceNorm1d(100)
        ).cuda()
        comp_module.load_state_dict(module.state_dict())

        sync_bn_module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)
        children = list(sync_bn_module.children())
        self.assertEqual(children[0].__class__, torch.nn.SyncBatchNorm)
        self.assertEqual(children[1].__class__, torch.nn.InstanceNorm1d)

        for layer, converted_layer in zip(comp_module.children(), sync_bn_module.children()):
            for key in layer.state_dict().keys():
                self.assertEqual(layer.state_dict()[key].device, converted_layer.state_dict()[key].device)
                self.assertEqual(layer.state_dict()[key], converted_layer.state_dict()[key])

    

    def _test_batchnorm_eval(self, ndim, device, dtype, module_dtype=None):
        module_dtype = module_dtype or dtype
        module = nn.BatchNorm1d(3).to(device, module_dtype)
        module.eval()

        data = torch.rand([3] * ndim, device=device, dtype=dtype, requires_grad=True)
        grad = torch.rand([3] * ndim, device=device, dtype=dtype)

        # 1st pass
        res1 = module(data)
        res1.backward(grad)
        grad1 = data.grad.clone()

        # 2nd pass
        if data.grad is not None:
            data.grad.data.zero_()

        res2 = module(data)
        res2.backward(grad)
        grad2 = data.grad.clone()
        self.assertEqual(res1, res2)
        self.assertEqual(grad1, grad2)

        # track_running_stats=False
        module = nn.BatchNorm1d(3, track_running_stats=False).to(device, module_dtype)

        data = torch.rand(4, 3, device=device, dtype=dtype, requires_grad=True)
        grad = torch.rand(4, 3, device=device, dtype=dtype)

        # 1st pass
        res1 = module(data)
        res1.backward(grad)
        grad1 = data.grad.clone()

        # set eval
        module.eval()

        # 2nd pass
        if data.grad is not None:
            data.grad.data.zero_()

        res2 = module(data)
        res2.backward(grad)
        grad2 = data.grad.clone()
        self.assertEqual(res1, res2)
        self.assertEqual(grad1, grad2)


    def _test_batchnorm_affine(self, ndim, device, dtype, module_dtype=None):
        # Compare affine against no-op weights and bias
        module_dtype = module_dtype or dtype
        module = nn.BatchNorm1d(3, affine=False).to(device, module_dtype)
        module_affine = nn.BatchNorm1d(3, affine=True).to(device, module_dtype)
        with torch.no_grad():
            module_affine.weight.fill_(1.0)
            module_affine.bias.zero_()

        data = torch.rand([3] * ndim, device=device, dtype=dtype, requires_grad=True)
        grad = torch.ones_like(data, requires_grad=False)

        # With weights all ones and bias all zeros
        res1 = module_affine(data)
        res1.backward(grad)
        grad1 = data.grad.clone()
        data.grad.zero_()

        # Without any weights or bias
        res2 = module(data)
        res2.backward(grad)
        grad2 = data.grad

        self.assertEqual(res1, res2)
        self.assertEqual(grad1, grad2)

    def _test_batchnorm_simple_average(self, device, dtype, module_dtype=None):
        module_dtype = module_dtype or dtype
        module = nn.BatchNorm1d(3, momentum=None).to(dtype=module_dtype, device=device)
        zeros = torch.zeros(3, dtype=module_dtype, device=device)
        ones = torch.ones(3, dtype=module_dtype, device=device)
        self.assertEqual(module.running_mean, zeros)
        self.assertEqual(module.running_var, ones)

        data1 = torch.rand(4, 3, dtype=dtype, device=device)
        data2 = torch.rand(4, 3, dtype=dtype, device=device)

        # 1st pass
        res1 = module(data1)
        running_mean1 = module.running_mean.clone()
        running_var1 = module.running_var.clone()
        self.assertNotEqual(running_mean1, zeros)
        self.assertNotEqual(running_var1, ones)

        # reset stats
        module.reset_running_stats()
        self.assertEqual(module.running_mean, zeros)
        self.assertEqual(module.running_var, ones)

        # 2nd pass
        res2 = module(data2)
        running_mean2 = module.running_mean.clone()
        running_var2 = module.running_var.clone()
        self.assertNotEqual(running_mean2, zeros)
        self.assertNotEqual(running_var2, ones)

        # reset stats
        module.reset_running_stats()
        self.assertEqual(module.running_mean, zeros)
        self.assertEqual(module.running_var, ones)

        # 3rd (combined) pass
        res3 = module(data1)
        res4 = module(data2)
        self.assertEqual(res3, res1)
        self.assertEqual(res4, res2)
        self.assertEqual(module.running_mean, (running_mean1 + running_mean2) / 2)
        self.assertEqual(module.running_var, (running_var1 + running_var2) / 2)

    
    def _test_batchnorm_update_stats(self, device, dtype=torch.float):
        module = nn.BatchNorm1d(3).to(device, dtype)

        data = torch.rand(4, 3, device=device, dtype=dtype)

        # training pass
        old_running_mean = module.running_mean.clone()
        old_running_var = module.running_var.clone()
        old_num_batches_tracked = module.num_batches_tracked.clone()
        module(data)
        self.assertNotEqual(old_running_mean, module.running_mean)
        self.assertNotEqual(old_running_var, module.running_var)
        self.assertEqual(old_num_batches_tracked + 1, module.num_batches_tracked)

        # eval pass
        module.eval()
        old_running_mean = module.running_mean.clone()
        old_running_var = module.running_var.clone()
        old_num_batches_tracked = module.num_batches_tracked.clone()
        module(data)
        self.assertEqual(old_running_mean, module.running_mean)
        self.assertEqual(old_running_var, module.running_var)
        self.assertEqual(old_num_batches_tracked, module.num_batches_tracked)

    
instantiate_parametrized_tests(TestNN)

if __name__ == '__main__':
    run_tests()
