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
    parametrize as parametrize_test, subtest, IS_WINDOWS
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU, TEST_CUDNN, TEST_CUDNN_VERSION
from torch.testing._internal.common_nn import NNTestCase, NewModuleTest, CriterionTest, \
    module_tests, criterion_tests, loss_reference_fns, \
    ctcloss_reference, new_module_tests, single_batch_reference_fn
from torch.testing._internal.common_device_type import dtypes, \
    dtypesIfCUDA, precisionOverride, skipCUDAIfCudnnVersionLessThan, onlyCUDA, onlyCPU, \
    skipCUDAIfRocm, skipCUDAIf, skipCUDAIfNotRocm, \
    onlyNativeDeviceTypes, deviceCountAtLeast, largeTensorTest, expectedFailureMeta, skipMeta, get_all_device_types
from torch.nn import MultiheadAttention

from hypothesis import given
import torch.testing._internal.hypothesis_utils as hu
from torch.testing._internal.common_utils import _assertGradAndGradgradChecks, gradcheck, gradgradcheck, \
    GRADCHECK_NONDET_TOL
from torch.testing._internal.common_utils import dtype2prec_DONTUSE
from torch.testing._internal.common_cuda import tf32_on_and_off, tf32_is_not_fp32, tf32_off, tf32_on
from torch.types import _TensorOrTensors

from ..utils.timer_wrapper import pytorch_op_timer
from ..utils.pytorch_device_decorators import onlyAcceleratedDeviceTypes, instantiate_device_type_tests


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


    def test_RNN_cell_no_broadcasting(self, device):
        def test(cell_module, input, hx, input_size, hidden_size):
            with pytorch_op_timer():
                cell = cell_module(input_size, hidden_size)
            self.assertRaises(RuntimeError, lambda: cell(input, hx))

        def test_all(hidden_size, bad_hx, good_hx, input_size, input):
            test(nn.LSTMCell, input, (bad_hx, good_hx), input_size, hidden_size)
            test(nn.LSTMCell, input, (good_hx, bad_hx), input_size, hidden_size)

        hidden_size = 20
        input_size = 10
        input = torch.randn(3, input_size, device=device)
        bad_hx = torch.randn(1, hidden_size, device=device)
        good_hx = torch.randn(3, hidden_size, device=device)

        # Test hidden/input batch size broadcasting
        test_all(hidden_size, bad_hx, good_hx, input_size, input)

        # Test hx's hidden_size vs module's hidden_size broadcasting
        bad_hx = torch.randn(3, 1, device=device)
        test_all(hidden_size, bad_hx, good_hx, input_size, input)

        # Test input's input_size vs module's input_size broadcasting
        bad_input = torch.randn(3, 1)
        test_all(hidden_size, good_hx, good_hx, input_size, bad_input)

    def test_LSTM_cell(self, device):
        # this is just a smoke test; these modules are implemented through
        # autograd so no Jacobian test is needed
        for bias in (True, False):
            input = torch.randn(3, 10, device=device)
            hx = torch.randn(3, 20, device=device)
            cx = torch.randn(3, 20, device=device)
            with pytorch_op_timer():
                lstm = nn.LSTMCell(10, 20, bias=bias, device=device)
            for _ in range(6):
                hx, cx = lstm(input, (hx, cx))

            (hx + cx).sum().backward()

    def test_LSTM_cell_forward_input_size(self, device):
        input = torch.randn(3, 11, device=device)
        hx = torch.randn(3, 20, device=device)
        cx = torch.randn(3, 20, device=device)
        with pytorch_op_timer():
            lstm = nn.LSTMCell(10, 20, device=device)
        self.assertRaises(Exception, lambda: lstm(input, (hx, cx)))

    def test_LSTM_cell_forward_hidden_size(self, device):
        input = torch.randn(3, 10, device=device)
        hx = torch.randn(3, 21, device=device)
        cx = torch.randn(3, 20, device=device)
        with pytorch_op_timer():
            lstm = nn.LSTMCell(10, 20, device=device)
        self.assertRaises(Exception, lambda: lstm(input, (hx, cx)))
        self.assertRaises(Exception, lambda: lstm(input, (cx, hx)))


    

class TestNNDeviceType(NNTestCase):
    @onlyAcceleratedDeviceTypes
    @dtypes(torch.double)
    def test_lstmcell_backward_only_one_output_grad(self, device, dtype):
        # checks that undefined gradients doen't hamper the backward
        # see #11872
        with pytorch_op_timer():
            l = torch.nn.LSTMCell(2, 3).to(device).to(dtype=dtype)
        s = torch.randn(1, 2, device=device, dtype=dtype, requires_grad=True)
        for i in range(2):
            out = l(s)[i]
            out.sum().backward()
            self.assertFalse(s.grad is None or s.grad.abs().sum().item() == 0)

    
instantiate_device_type_tests(TestNNDeviceType, globals())
instantiate_device_type_tests(TestNN, globals())

if __name__ == '__main__':
    run_tests()
