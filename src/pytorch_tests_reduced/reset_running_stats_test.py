# Owner(s): ["module: nn"]

from torch.types import _TensorOrTensors
from torch.testing._internal.common_cuda import tf32_on_and_off, tf32_is_not_fp32, tf32_off, tf32_on
from torch.testing._internal.common_utils import dtype2prec_DONTUSE
from torch.testing._internal.common_utils import _assertGradAndGradgradChecks, gradcheck, gradgradcheck, \
    GRADCHECK_NONDET_TOL
import torch.testing._internal.hypothesis_utils as hu
from hypothesis import given
from torch.testing._internal.common_device_type import dtypes, \
    dtypesIfCUDA, precisionOverride, skipCUDAIfCudnnVersionLessThan, onlyCUDA, onlyCPU, \
    skipCUDAIfRocm, skipCUDAIf, skipCUDAIfNotRocm, \
    onlyNativeDeviceTypes, deviceCountAtLeast, largeTensorTest, expectedFailureMeta, skipMeta, get_all_device_types
from torch.testing._internal.common_nn import NNTestCase, NewModuleTest, CriterionTest
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU, TEST_CUDNN, TEST_CUDNN_VERSION
from torch.testing._internal.common_utils import freeze_rng_state, run_tests, TestCase, skipIfNoLapack, skipIfRocm, \
    TEST_NUMPY, TEST_SCIPY, TEST_WITH_CROSSREF, TEST_WITH_ROCM, \
    download_file, get_function_arglist, load_tests, skipIfMps,\
    TEST_WITH_UBSAN, IS_PPC, \
    parametrize as parametrize_test, subtest, instantiate_parametrized_tests
from torch.testing._internal.common_dtype import integral_types, get_all_math_dtypes
from torch.nn.parallel._functions import Broadcast
from torch.nn import Parameter
from torch.nn.utils.fusion import fuse_linear_bn_weights
from torch.nn.utils.fusion import fuse_conv_bn_weights
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.autograd.forward_ad as fwAD
from torch._six import inf, nan
import contextlib
import math
import random
import unittest
import io
import itertools
import warnings
import pickle
from copy import deepcopy
from itertools import product
from functools import partial
from collections import OrderedDict

import torch
from ..utils.pytorch_device_decorators import onlyNativeDeviceTypes, onlyAcceleratedDeviceTypes, instantiate_device_type_tests
from ..utils.timer_wrapper import pytorch_op_timer
# TODO: remove this global setting
# NN tests use double as the default dtype
torch.set_default_dtype(torch.double)


AMPERE_OR_ROCM = TEST_WITH_ROCM or tf32_is_not_fp32()

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

if TEST_SCIPY:
    import scipy.signal
    import scipy.ndimage

if TEST_NUMPY:
    import numpy as np


# WARNING: If you add a new top-level test case to this file, you MUST
# update test/run_test.py to list it, otherwise it will NOT be run in
# CI.

class TestNNDeviceType(NNTestCase):
    def _test_batchnorm_simple_average(self, device, dtype, module_dtype=None):
        module_dtype = module_dtype or dtype
        module = nn.BatchNorm1d(3, momentum=None).to(
            dtype=module_dtype, device=device)
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
        with pytorch_op_timer():
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
        with pytorch_op_timer():
            module.reset_running_stats()
        self.assertEqual(module.running_mean, zeros)
        self.assertEqual(module.running_var, ones)

        # 3rd (combined) pass
        res3 = module(data1)
        res4 = module(data2)
        self.assertEqual(res3, res1)
        self.assertEqual(res4, res2)
        self.assertEqual(module.running_mean,
                         (running_mean1 + running_mean2) / 2)
        self.assertEqual(module.running_var, (running_var1 + running_var2) / 2)

    @dtypes(torch.float, torch.bfloat16)
    # @dtypesIfCUDA(torch.float, torch.bfloat16)
    def test_batchnorm_simple_average(self, device, dtype):
        self._test_batchnorm_simple_average(device, dtype)

        # if self.device_type == 'cuda' and self.has_cudnn():
        #     with torch.backends.cudnn.flags(enabled=False):
        #         self._test_batchnorm_simple_average(device, dtype)


instantiate_device_type_tests(TestNNDeviceType, globals())

if __name__ == '__main__':
    run_tests()
