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
from ..utils.pytorch_device_decorators import onlyNativeDeviceTypes, onlyAcceleratedDeviceTypes, instantiate_device_type_tests

# TODO: remove this global setting
# NN tests use double as the default dtype
torch.set_default_dtype(torch.double)


class TestNN(NNTestCase):
    def test_Sequential_getitem(self, device):
        l1 = nn.Linear(10, 20).to(device)
        l2 = nn.Linear(20, 30).to(device)
        l3 = nn.Linear(30, 40).to(device)
        l4 = nn.Linear(40, 50).to(device)
        n = nn.Sequential(l1, l2, l3, l4).to(device)
        self.assertIs(n[0], l1)
        self.assertIs(n[1], l2)
        self.assertIs(n[2], l3)
        self.assertIs(n[3], l4)
        self.assertIs(n[torch.tensor(3, dtype=torch.int64, device=device)], l4)
        self.assertEqual(n[1:], nn.Sequential(l2, l3, l4).to(device))
        self.assertEqual(n[3:], nn.Sequential(l4).to(device))
        self.assertEqual(n[:-1], nn.Sequential(l1, l2, l3).to(device))
        self.assertEqual(n[:-3], nn.Sequential(l1).to(device))
        self.assertEqual(n[::-1], nn.Sequential(l4, l3, l2, l1).to(device))

    def test_Sequential_setitem(self, device):
        l1 = nn.Linear(10, 20).to(device)
        l2 = nn.Linear(20, 30).to(device)
        l3 = nn.Linear(30, 40).to(device)
        l4 = nn.Linear(40, 50).to(device)
        n = nn.Sequential(l1, l2, l3).to(device)
        n[0] = l4
        n[-1] = l4
        n[torch.tensor(1, dtype=torch.int16, device=device)] = l1
        self.assertIs(n[0], l4)
        self.assertIs(n[1], l1)
        self.assertIs(n[2], l4)

    def test_Sequential_setitem_named(self, device):
        l1 = nn.Linear(10, 20).to(device)
        l2 = nn.Linear(20, 30).to(device)
        l3 = nn.Linear(30, 40).to(device)
        l4 = nn.Linear(40, 50).to(device)
        n = nn.Sequential(OrderedDict([
            ('linear1', l1),
            ('linear2', l2),
            ('linear3', l3),
        ])).to(device)

        n[0] = l4
        n[-1] = l4
        self.assertEqual(n.linear1, l4)
        self.assertEqual(n.linear3, l4)

    def test_Sequential_delitem(self, device):
        l1 = nn.Linear(10, 20).to(device)
        l2 = nn.Linear(20, 30).to(device)
        l3 = nn.Linear(30, 40).to(device)
        l4 = nn.Linear(40, 50).to(device)
        n = nn.Sequential(l1, l2, l3, l4).to(device)
        del n[-1]
        self.assertEqual(n, nn.Sequential(l1, l2, l3).to(device))
        del n[1::2]
        self.assertEqual(n, nn.Sequential(l1, l3).to(device))

    def test_Sequential_append(self, device):
        l1 = nn.Linear(10, 20).to(device)
        l2 = nn.Linear(20, 30).to(device)
        l3 = nn.Linear(30, 40).to(device)
        l4 = nn.Linear(40, 50).to(device)
        n = nn.Sequential(l1, l2, l3).to(device)
        n2 = n.append(l4)
        self.assertEqual(n, nn.Sequential(l1, l2, l3, l4).to(device))
        self.assertEqual(n2, nn.Sequential(l1, l2, l3, l4).to(device))
        self.assertEqual(nn.Sequential(l1).append(
            l2).append(l4).to(device), nn.Sequential(l1, l2, l4).to(device))


instantiate_device_type_tests(TestNN, globals())

if __name__ == '__main__':
    run_tests()
