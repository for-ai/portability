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


class TestNN(NNTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    def test_Transformer_cell(self):
        # this is just a smoke test; these modules are implemented through
        # autograd so no Jacobian test is needed
        d_model = 512
        nhead = 16
        num_encoder_layers = 4
        num_decoder_layers = 3
        dim_feedforward = 256
        dropout = 0.3
        bsz = 8
        seq_length = 35
        tgt_length = 15
        for batch_first, src_size, tgt_size in zip((True, False),
                                                   [(bsz, seq_length, d_model),
                                                    (seq_length, bsz, d_model)],
                                                   [(bsz, tgt_length, d_model),
                                                    (tgt_length, bsz, d_model)]):
            transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers,
                                         dim_feedforward, dropout, batch_first=batch_first)
            src = torch.randn(src_size)
            src_mask = transformer.generate_square_subsequent_mask(
                seq_length).double()
            tgt = torch.randn(tgt_size)
            tgt_mask = transformer.generate_square_subsequent_mask(
                tgt_length).double()
            memory_mask = torch.randn(tgt_length, seq_length).double()
            src_key_padding_mask = torch.rand(bsz, seq_length) >= 0.5
            tgt_key_padding_mask = torch.rand(bsz, tgt_length) >= 0.5
            memory_key_padding_mask = torch.rand(bsz, seq_length) >= 0.5

            output = transformer(src, tgt,
                                 src_mask=src_mask,
                                 tgt_mask=tgt_mask,
                                 memory_mask=memory_mask,
                                 src_key_padding_mask=src_key_padding_mask,
                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                 memory_key_padding_mask=memory_key_padding_mask)
            output.sum().backward()


instantiate_parametrized_tests(TestNN)

if __name__ == '__main__':
    run_tests()
