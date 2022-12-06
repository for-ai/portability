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
    def _test_InstanceNorm_general(self, cls, input, device, dtype=torch.float):
        # default case track_running_stats=False
        b, c = input.size(0), input.size(1)
        input_var = input.to(device=device, dtype=dtype).requires_grad_()

        IN = cls(c, eps=0).to(device, dtype)

        output = IN(input_var)
        out_reshaped = output.view(b * c, -1)

        mean = out_reshaped.mean(1)
        var = out_reshaped.var(1, unbiased=False)

        self.assertEqual(torch.abs(mean.data).mean(), 0, atol=1e-5, rtol=0)
        self.assertEqual(torch.abs(var.data).mean(), 1, atol=1e-5, rtol=0)

        # check that eval mode doesn't change behavior
        grad_out = torch.randn_like(output)
        res1 = output.data.clone()
        output.backward(grad_out)
        grad1 = input_var.grad.data.clone()

        IN.eval()
        output = IN(input_var)
        input_var.grad = None
        output.backward(grad_out)
        res2 = output.data
        grad2 = input_var.grad.data
        self.assertEqual(res1, res2)
        self.assertEqual(grad1, grad2)

        # If track_running_stats=True and momentum=1, running_mean/var should be
        # equal to mean/var of the input (with unbias correction)
        IN = cls(c, momentum=1, eps=0,
                 track_running_stats=True).to(device, dtype)

        output = IN(input_var)

        input_reshaped = input_var.transpose(1, 0).reshape(c, -1)
        mean = input_reshaped.mean(1)

        input_reshaped = input_var.transpose(1, 0).reshape(c, b, -1)
        var = input_reshaped.var(2, unbiased=True)[:, :]

        self.assertEqual(
            torch.abs(mean.data - IN.running_mean).mean(), 0, atol=1e-5, rtol=0)
        self.assertEqual(torch.abs(var.data.mean(
            1) - IN.running_var).mean(), 0, atol=1e-5, rtol=0)

        # in eval mode, adding X * std to a channel in input should make the
        # corresponding channel in output have mean X
        IN.eval()
        delta = IN.running_var.sqrt() * torch.arange(c, device=device, dtype=dtype)
        delta = delta.view(-1, *[1 for _ in range(2, input.dim())])
        output = IN(input_var + delta)
        self.assertEqual(output.transpose(0, 1).reshape(
            c, -1).mean(1), torch.arange(c, dtype=dtype))

    def test_InstanceNorm2d_general(self, device):
        b = random.randint(3, 5)
        c = random.randint(3, 5)
        w = random.randint(3, 6)
        h = random.randint(6, 8)

        input = torch.rand(b, c, h, w)
        self._test_InstanceNorm_general(nn.InstanceNorm2d, input, device)

        if self.device_type == 'cuda':
            self._test_InstanceNorm_cuda_half(nn.InstanceNorm2d, input, device)


instantiate_device_type_tests(TestNNDeviceType, globals())

if __name__ == '__main__':
    run_tests()
