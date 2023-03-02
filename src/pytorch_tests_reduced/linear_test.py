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
    def test_to(self, device):
        m = nn.Linear(3, 5).to(device)
        self.assertIs(m, m.to('cpu'))
        self.assertIs(m, m.to('cpu', dtype=torch.float32))
        self.assertEqual(m.double(), m.to(torch.float64))
        self.assertRaises(RuntimeError, lambda: m.to('cpu', copy=True))

        # if torch.cuda.is_available():
        #     for cuda in ['cuda', 'cuda:0' if torch.cuda.device_count() == 1 else 'cuda:1']:
        #         m2 = m.cuda(device=cuda)
        #         self.assertIs(m2, m2.to(cuda))
        #         self.assertEqual(m, m2.to('cpu'))
        #         self.assertEqual(m2, m.to(cuda))
        #         self.assertIs(m2, m2.to(dtype=torch.float32))
        #         self.assertEqual(m2.double(), m2.to(dtype=torch.float64))


instantiate_device_type_tests(TestNN, globals())

if __name__ == '__main__':
    run_tests()
