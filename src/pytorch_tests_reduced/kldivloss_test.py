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
from torch.testing._internal.common_device_type import dtypes, \
    dtypesIfCUDA, precisionOverride, skipCUDAIfCudnnVersionLessThan, onlyCUDA, onlyCPU, \
    skipCUDAIfRocm, skipCUDAIf, skipCUDAIfNotRocm, \
    deviceCountAtLeast, largeTensorTest, expectedFailureMeta, skipMeta, get_all_device_types
from torch.nn import MultiheadAttention

from hypothesis import given
import torch.testing._internal.hypothesis_utils as hu
from torch.testing._internal.common_utils import _assertGradAndGradgradChecks, gradcheck, gradgradcheck, \
    GRADCHECK_NONDET_TOL
from torch.testing._internal.common_utils import dtype2prec_DONTUSE
from torch.testing._internal.common_cuda import tf32_on_and_off, tf32_is_not_fp32, tf32_off, tf32_on
from torch.types import _TensorOrTensors
from ..utils.pytorch_device_decorators import onlyNativeDeviceTypes, onlyAcceleratedDeviceTypes, instantiate_device_type_tests
from ..utils.timer_wrapper import pytorch_op_timer

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

    def test_KLDivLoss_batch_mean(self, device):
        torch.set_default_dtype(torch.double)
        input_shape = (2, 5)
        log_prob1 = F.log_softmax(torch.randn(input_shape, device=device), 1).to(device)
        prob2 = F.softmax(torch.randn(input_shape, device=device), 1).to(device)
        with pytorch_op_timer():
            loss = nn.KLDivLoss(reduction='batchmean')
        l = loss(log_prob1, prob2)
        with pytorch_op_timer():
            loss_none_reduce = nn.KLDivLoss(reduction='sum')(log_prob1, prob2)
        expected = loss_none_reduce / input_shape[0]

        self.assertEqual(l, expected)
        torch.set_default_dtype(torch.float)

    def test_KLDivLoss_batch_mean_log_target(self, device):
        torch.set_default_dtype(torch.double)
        input_shape = (2, 5)
        log_prob1 = F.log_softmax(torch.randn(input_shape, device=device), 1).to(device)
        log_prob2 = F.log_softmax(torch.randn(input_shape, device=device), 1).to(device)
        with pytorch_op_timer():
            loss = nn.KLDivLoss(reduction='batchmean', log_target=True)
        l = loss(log_prob1, log_prob2)
        with pytorch_op_timer():
            loss_none_reduce = nn.KLDivLoss(reduction='sum', log_target=True)(log_prob1, log_prob2)
        expected = loss_none_reduce / input_shape[0]

        self.assertEqual(l, expected)
        torch.set_default_dtype(torch.float)

instantiate_device_type_tests(TestNN, globals())

if __name__ == '__main__':
    run_tests()
