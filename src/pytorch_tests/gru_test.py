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

from hypothesis import given
import torch.testing._internal.hypothesis_utils as hu
from torch.testing._internal.common_utils import _assertGradAndGradgradChecks, gradcheck, gradgradcheck, \
    GRADCHECK_NONDET_TOL
from torch.testing._internal.common_utils import dtype2prec_DONTUSE
from torch.testing._internal.common_cuda import tf32_on_and_off, tf32_is_not_fp32, tf32_off, tf32_on
from torch.types import _TensorOrTensors


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

    @unittest.skipIf(not TEST_CUDNN, 'CUDNN not available')
    @skipIfRocm
    def test_cudnn_weight_format(self):
        rnns = [
            nn.GRU(10, 20, batch_first=True),
        ]
        first_warn = True
        for rnn in rnns:
            rnn.cuda()
            input = torch.randn(5, 4, 10, requires_grad=True, device="cuda")
            hx = torch.randn(1, 5, 20, requires_grad=True, device="cuda")
            all_vars = [input, hx] + list(rnn.parameters())
            if isinstance(rnn, nn.LSTM):
                # LSTM with projections has different hx size
                if rnn.proj_size > 0:
                    hx = torch.randn(1, 5, 10, requires_grad=True, device="cuda")
                    all_vars[1] = hx
                cx = torch.randn(1, 5, 20, requires_grad=True, device="cuda")
                all_vars[2:2] = [cx]
                hx = (hx, cx)

            output = rnn(input, hx)
            output[0].sum().backward()
            grads = [v.grad.data.clone() for v in all_vars]
            for v in all_vars:
                v.grad.data.zero_()

            # Weights will no longer view onto the same chunk of memory
            weight = all_vars[4]
            weight_data = weight.data.clone()
            with torch.no_grad():
                weight.set_(weight_data)

            for _ in range(2):
                with warnings.catch_warnings(record=True) as w:
                    output_noncontig = rnn(input, hx)
                if first_warn:
                    self.assertEqual(len(w), 1)
                    self.assertIn('weights are not part of single contiguous chunk of memory', w[0].message.args[0])
                    first_warn = False
                    warnings.resetwarnings()
                output_noncontig[0].sum().backward()
                grads_noncontig = [v.grad.data.clone() for v in all_vars]
                for v in all_vars:
                    v.grad.data.zero_()
                self.assertEqual(output, output_noncontig)
                self.assertEqual(grads_noncontig, grads)

            # Make sure these still share storage
            weight_data[:] = 4
            self.assertEqual(weight_data, all_vars[4].data)

    @unittest.skipIf(not TEST_CUDNN, 'CUDNN not available')
    def test_cudnn_weight_tying(self):
        rnns = [
    
            nn.GRU(10, 20, batch_first=True, bidirectional=True),
        ]
        for rnn in rnns:
            rnn.bias_ih_l0_reverse = rnn.bias_ih_l0
            rnn.cuda()
            input = torch.randn(5, 4, 10, requires_grad=True, device="cuda")
            hx = torch.randn(2, 5, 20, requires_grad=True, device="cuda")
            all_vars = [input, hx] + list(rnn.parameters())
            opt = torch.optim.SGD(rnn.parameters(), lr=0.1)
            opt.zero_grad()
            if isinstance(rnn, nn.LSTM):
                # LSTM with projections has different hx size
                if rnn.proj_size > 0:
                    hx = torch.randn(2, 5, 10, requires_grad=True, device="cuda")
                    all_vars[1] = hx
                cx = torch.randn(2, 5, 20, requires_grad=True, device="cuda")
                all_vars[2:2] = [cx]
                hx = (hx, cx)

            with warnings.catch_warnings(record=True) as w:
                output = rnn(input, hx)
            output[0].sum().backward()

            opt.step()
            with warnings.catch_warnings(record=True) as w:
                output_cuda = rnn(input, hx)
            rnn.cpu()
            hx = (hx[0].cpu(), hx[1].cpu()) if isinstance(rnn, nn.LSTM) else hx.cpu()
            output_cpu = rnn(input.cpu(), hx)
            self.assertEqual(output_cuda, output_cpu)

    def test_rnn_args_check(self):
        input_size = 3
        hidden_size = 5
        num_layers = 2
        batch_size = 4
        seq_len = 6
        num_directions = 1
        bad_size = 7  # prime number so that no size can divide it.

        def test(input_shape, hidden_shape, mode):
            for input, hidden in get_inputs(input_shape, hidden_shape, mode):
                model = getattr(nn, mode)(input_size, hidden_size, num_layers)
                self.assertRaises(RuntimeError, lambda: model(input, hidden))

        correct_input_shape = (seq_len, batch_size, input_size)
        correct_hidden_shape = (num_layers * num_directions, batch_size, hidden_size)

        def update_shape(shape, dim, new_dim_size):
            new_shape = list(shape)
            new_shape[dim] = new_dim_size
            return tuple(new_shape)

        def get_inputs(input_shape, hidden_shape, mode):
            '''returns list( tuple(input, hidden) )
            where input, hidden are inputs to a model'''
            input = torch.randn(input_shape)
            hidden = torch.randn(hidden_shape)
            if mode != 'LSTM':
                return [(input, hidden)]
            if hidden_shape == correct_hidden_shape:
                return [(input, (hidden, hidden))]
            good_hidden = torch.randn(correct_hidden_shape)
            return [
                (input, (hidden, good_hidden)),
                (input, (good_hidden, hidden)),
            ]

        rnn_modes = ['GRU']
        for mode in rnn_modes:
            # Incorrect input batch size
            input_shape = update_shape(correct_input_shape, 1, bad_size)
            hidden_shape = correct_hidden_shape
            test(input_shape, hidden_shape, mode)

            # Incorrect hidden batch size
            input_shape = correct_input_shape
            hidden_shape = update_shape(correct_hidden_shape, 1, bad_size)
            test(input_shape, hidden_shape, mode)

            # Incorrect input size
            input_shape = update_shape(correct_input_shape, 2, bad_size)
            hidden_shape = correct_hidden_shape
            test(input_shape, hidden_shape, mode)

            # Incorrect hidden size
            input_shape = correct_input_shape
            hidden_shape = update_shape(correct_hidden_shape, 2, bad_size)
            test(input_shape, hidden_shape, mode)

            # Incorrect hidden[0]
            input_shape = correct_input_shape
            hidden_shape = update_shape(correct_hidden_shape, 0, bad_size)
            test(input_shape, hidden_shape, mode)


    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_rnn_check_device(self):
        input_size = 3
        hidden_size = 5
        num_layers = 2
        batch_size = 4
        seq_len = 6
        num_directions = 1

        correct_input_shape = (seq_len, batch_size, input_size)
        correct_hidden_shape = (num_layers * num_directions, batch_size, hidden_size)
        rnn_modes = ['GRU']

        for mode in rnn_modes:
            model = getattr(nn, mode)(input_size, hidden_size, num_layers)
            input = torch.randn(correct_input_shape)
            hidden = torch.randn(correct_hidden_shape)

            # input and weights are not at the same device
            with self.assertRaisesRegex(RuntimeError,
                                        "Input and parameter tensors are not at the same device"):
                model(input.to('cuda:0'))

            # input and hiddens are not at the same device
            with self.assertRaisesRegex(RuntimeError,
                                        r"Input and hidden tensors are not at the same device"):
                if mode == 'LSTM':
                    model(input, (hidden.to('cuda:0'), hidden.to('cuda:0')))
                else:
                    model(input, (hidden.to('cuda:0')))

            # hidden tensors are not at the same CUDA device
            if mode == 'LSTM':
                with self.assertRaisesRegex(RuntimeError,
                                            "Input and hidden tensors are not at the same device"):
                    model(input.to('cuda:0'), (hidden.to('cuda:0'), hidden.to('cuda:1')))

    

    def test_rnn_initial_hidden_state(self):
        rnn_modes = [ 'GRU']
        for mode in rnn_modes:
            rnn = getattr(nn, mode)(30, 20, 2)
            input = torch.randn(10, 32, 30)
            hidden = torch.zeros(2, 32, 20)

            if mode == 'LSTM':
                hidden = (hidden, hidden)
            output1, hidden1 = rnn(input, hidden)
            output2, hidden2 = rnn(input)
            self.assertEqual(output1, output2)
            self.assertEqual(hidden1, hidden2)

    def test_projections_errors_on_gru_and_rnn(self):
        error_msg = "proj_size argument is only supported for LSTM, not RNN or GRU"
        for mode in ['GRU']:
            with self.assertRaisesRegex(ValueError, error_msg):
                rnn = getattr(nn, mode)(30, 20, 2, proj_size=10)

    def _test_RNN_cpu_vs_cudnn(self, dropout, dtype=torch.double):

        def forward_backward(cuda, rnn, input_val, grad_output, weights_val, hx_val, grad_hy,
                             cx_val=None, grad_cy=None):
            is_lstm = isinstance(rnn, nn.LSTM)

            for x_layer, y_layer in zip(rnn.all_weights, weights_val):
                for x, y in zip(x_layer, y_layer):
                    x.data.copy_(y.data)

            if isinstance(input_val, rnn_utils.PackedSequence):
                input = rnn_utils.PackedSequence(
                    input_val.data.data.requires_grad_(True), input_val.batch_sizes)
                input_var = input.data
            else:
                input = input_val.clone().requires_grad_(True)
                input_var = input
            if is_lstm:
                if cx_val is None:
                    hx = (hx_val.clone().requires_grad_(True),
                          hx_val.add(1).requires_grad_(True))
                else:
                    hx = (hx_val.clone().requires_grad_(True),
                          cx_val.add(1).requires_grad_(True))
            else:
                hx = hx_val.clone().requires_grad_(True)

            if cuda:
                rnn.cuda()
                input_var.data = input_var.data.cuda()
                if is_lstm:
                    hx[0].data = hx[0].data.cuda()
                    hx[1].data = hx[1].data.cuda()
                else:
                    hx.data = hx.data.cuda()
                grad_hy = grad_hy.cuda()
                if grad_cy is not None:
                    grad_cy = grad_cy.cuda()
                grad_output = grad_output.cuda()

            output, hy = rnn(input, hx)

            if isinstance(output, rnn_utils.PackedSequence):
                output = output.data

            if is_lstm:
                if grad_cy is None:
                    torch.autograd.backward([output, hy[0], hy[1]], [grad_output, grad_hy, grad_hy + 1])
                else:
                    torch.autograd.backward([output, hy[0], hy[1]], [grad_output, grad_hy, grad_cy + 1])
            else:
                torch.autograd.backward([output, hy], [grad_output, grad_hy])

            return {'output': output.data,
                    'hy': hy[0].data if is_lstm else hy.data,
                    'weights': rnn.all_weights,
                    'grad_input': input_var.grad.data,
                    'grad_hx': hx[0].grad.data if is_lstm else hx.grad.data,
                    'cy': hy[1].data if is_lstm else None,
                    'grad_cx': hx[1].grad.data if is_lstm else None}

        input_size = 10
        hidden_size = 6
        proj_size = 3
        num_layers = 2
        seq_length = 7
        batch = 6

        def make_noncontig(tensor):
            ndim = tensor.dim()
            return torch.stack([tensor.clone().zero_(), tensor], ndim).select(ndim, 1)

        def compare_cpu_gpu(outputs_cpu, outputs_gpu):
            self.assertEqual(list(outputs_cpu.keys()), list(outputs_gpu.keys()))
            for key in outputs_cpu.keys():
                if key != 'weights':
                    self.assertEqual(outputs_cpu[key], outputs_gpu[key], atol=5e-5, rtol=0, msg=key)

            # check grad weights separately, as nested dict
            for cpu_layer_weight, gpu_layer_weight in zip(outputs_cpu['weights'], outputs_gpu['weights']):
                for (cpu_weight, gpu_weight) in zip(cpu_layer_weight, gpu_layer_weight):
                    self.assertEqual(cpu_weight.grad.data, gpu_weight.grad.data, atol=5e-5, rtol=0)

        for module in (nn.GRU):
            for bias, bidirectional, batch_first, contig, variable_len, lens_as_tensor \
                    in product((True, False), repeat=6):

                num_directions = 2 if bidirectional else 1
                if batch_first:
                    input_val = torch.randn(batch, seq_length, input_size, dtype=dtype)
                    grad_output = torch.randn(batch, seq_length, hidden_size * num_directions, dtype=dtype)
                else:
                    input_val = torch.randn(seq_length, batch, input_size, dtype=dtype)
                    grad_output = torch.randn(seq_length, batch, hidden_size * num_directions, dtype=dtype)

                hx_val = torch.randn(num_layers * num_directions, batch, hidden_size, dtype=dtype)
                grad_hy = torch.randn(num_layers * num_directions, batch, hidden_size, dtype=dtype)

                if not contig:
                    grad_output = make_noncontig(grad_output)
                    grad_hy = make_noncontig(grad_hy)
                    input_var = make_noncontig(input_val)
                    hx_val = make_noncontig(hx_val)

                if variable_len:
                    lengths = [7, 5, 5, 2, 1, 1]
                    if lens_as_tensor:
                        lengths = torch.tensor(lengths, dtype=torch.long)
                    input_val = rnn_utils.pack_padded_sequence(input_val, lengths, batch_first=batch_first)
                    grad_output = rnn_utils.pack_padded_sequence(grad_output, lengths, batch_first=batch_first).data

                rnn = module(input_size,
                             hidden_size,
                             num_layers,
                             bias=bias,
                             dropout=dropout,
                             bidirectional=bidirectional,
                             batch_first=batch_first).to(dtype)

                outputs_cpu = forward_backward(
                    False, rnn, input_val, grad_output, rnn.all_weights, hx_val, grad_hy)

                rnn_gpu = module(input_size,
                                 hidden_size,
                                 num_layers,
                                 bias=bias,
                                 dropout=dropout,
                                 bidirectional=bidirectional,
                                 batch_first=batch_first).to(dtype)

                outputs_gpu = forward_backward(
                    True, rnn_gpu, input_val, grad_output, rnn.all_weights, hx_val, grad_hy)

                compare_cpu_gpu(outputs_cpu, outputs_gpu)

        for nonlinearity in ('tanh', 'relu'):
            hx_val = torch.randn(num_layers, batch, hidden_size, dtype=dtype)
            input_val = torch.randn(seq_length, batch, input_size, dtype=dtype)
            grad_output = torch.randn(
                seq_length, batch, hidden_size * num_directions, dtype=dtype)
            grad_hy = torch.randn(
                num_layers * num_directions, batch, hidden_size, dtype=dtype)

            rnn = nn.RNN(input_size, hidden_size, num_layers, bias=bias, nonlinearity=nonlinearity).to(dtype)
            outputs_cpu = forward_backward(False, rnn, input_val, grad_output, rnn.all_weights, hx_val, grad_hy)

            rnn_gpu = nn.RNN(input_size, hidden_size, num_layers, bias=bias, nonlinearity=nonlinearity).to(dtype)
            outputs_gpu = forward_backward(True, rnn_gpu, input_val, grad_output, rnn.all_weights, hx_val, grad_hy)

            compare_cpu_gpu(outputs_cpu, outputs_gpu)

        # checking LSTM with projections
        for bias, bidirectional, batch_first, contig, variable_len, lens_as_tensor \
                in product((True, False), repeat=6):
            num_directions = 2 if bidirectional else 1
            if batch_first:
                input_val = torch.randn(batch, seq_length, input_size, dtype=dtype)
                grad_output = torch.randn(batch, seq_length, proj_size * num_directions, dtype=dtype)
            else:
                input_val = torch.randn(seq_length, batch, input_size, dtype=dtype)
                grad_output = torch.randn(seq_length, batch, proj_size * num_directions, dtype=dtype)

            hx_val = torch.randn(num_layers * num_directions, batch, proj_size, dtype=dtype)
            cx_val = torch.randn(num_layers * num_directions, batch, hidden_size, dtype=dtype)
            grad_hy = torch.randn(num_layers * num_directions, batch, proj_size, dtype=dtype)
            grad_cy = torch.randn(num_layers * num_directions, batch, hidden_size, dtype=dtype)

            if not contig:
                grad_output = make_noncontig(grad_output)
                grad_hy = make_noncontig(grad_hy)
                grad_cy = make_noncontig(grad_cy)
                input_var = make_noncontig(input_val)
                hx_val = make_noncontig(hx_val)
                cx_val = make_noncontig(cx_val)

            if variable_len:
                lengths = [7, 5, 5, 2, 1, 1]
                if lens_as_tensor:
                    lengths = torch.tensor(lengths, dtype=torch.long)
                input_val = rnn_utils.pack_padded_sequence(input_val, lengths, batch_first=batch_first)
                grad_output = rnn_utils.pack_padded_sequence(grad_output, lengths, batch_first=batch_first).data

            rnn = nn.LSTM(input_size,
                          hidden_size,
                          num_layers,
                          bias=bias,
                          dropout=dropout,
                          bidirectional=bidirectional,
                          batch_first=batch_first,
                          proj_size=proj_size).to(dtype)

            outputs_cpu = forward_backward(
                False, rnn, input_val, grad_output, rnn.all_weights,
                hx_val, grad_hy, cx_val, grad_cy)

            rnn_gpu = nn.LSTM(input_size,
                              hidden_size,
                              num_layers,
                              bias=bias,
                              dropout=dropout,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              proj_size=proj_size).to(dtype)

            outputs_gpu = forward_backward(
                True, rnn_gpu, input_val, grad_output, rnn.all_weights,
                hx_val, grad_hy, cx_val, grad_cy)
            compare_cpu_gpu(outputs_cpu, outputs_gpu)

    @unittest.skipIf(not TEST_CUDNN, "needs cudnn")
    def test_RNN_cpu_vs_cudnn_no_dropout(self):
        dtype = torch.double
        self._test_RNN_cpu_vs_cudnn(0, dtype)

    @unittest.skipIf(not (TEST_CUDNN and (TEST_CUDNN_VERSION if TEST_CUDNN_VERSION else 0) >= 5103), "needs cudnn >= 5.1")
    def test_RNN_cpu_vs_cudnn_with_dropout(self):
        # Because of dropout randomness, can only compare dropout=0 and dropout=1
        self._test_RNN_cpu_vs_cudnn(1)

   
    def test_error_RNN_seq_len_zero(self):
        # checking error message when RNN has seq_len = 0
        for module in [nn.GRU]:
            for bidirectional in [True, False]:
                for device in get_all_device_types():
                    input = torch.ones(0, 10, 5)
                    rnn = module(5, 6, bidirectional=bidirectional)
                    if device == 'cuda':
                        rnn.cuda()
                        input = input.cuda()

                    with self.assertRaisesRegex(RuntimeError, "Expected sequence length to be larger than 0 in RNN"):
                        rnn(input)

    def test_RNN_input_size_zero(self):
        for module in [nn.GRU]:
            for device in get_all_device_types():
                input = torch.zeros((5, 0, 3))
                rnn = module(input_size=3, hidden_size=4)
                if device == 'cuda':
                    rnn.cuda()
                    input = input.cuda()
                outs = rnn(input)
                self.assertEqual(outs[0].shape, torch.Size([5, 0, 4]))
                # Check that backward does not cause a hard error
                outs[0].sum().backward()

    

class TestNNDeviceType(NNTestCase):
    @onlyCUDA
    @dtypes(torch.float, torch.double)
    @tf32_on_and_off(0.005)
    def test_rnn_fused(self, device, dtype):

        def copy_rnn(rnn1, rnn2):
            for x_layer, y_layer in zip(rnn1.all_weights, rnn2.all_weights):
                for x, y in zip(x_layer, y_layer):
                    x.data.copy_(y.data)

        def check_rnn_grads(rnn1, rnn2):
            for x_layer, y_layer in zip(rnn1.all_weights, rnn2.all_weights):
                for x, y in zip(x_layer, y_layer):
                    self.assertEqual(x.grad, y.grad, atol=5e-5, rtol=0)

        input_size = 10
        hidden_size = 6
        num_layers = 2
        seq_length = 7
        batch = 6
        input_val = torch.randn(seq_length, batch, input_size, dtype=dtype)
        grad_output = torch.randn(seq_length, batch, hidden_size, dtype=dtype)
        hx_val = torch.randn(num_layers, batch, hidden_size, dtype=dtype)
        grad_hy = torch.randn(num_layers, batch, hidden_size, dtype=dtype)
        with torch.backends.cudnn.flags(enabled=False, allow_tf32=None):
            for module in (nn.GRU):
                for bias in (True, False):
                    rnn = module(input_size, hidden_size, num_layers, bias=bias).to(dtype)
                    rnn_device = module(input_size, hidden_size, num_layers, bias=bias).to(device, dtype)
                    copy_rnn(rnn, rnn_device)

                    is_lstm = isinstance(rnn, nn.LSTM)
                    if is_lstm:
                        hx = (hx_val.clone().requires_grad_(True),
                              hx_val.clone().add(1).requires_grad_(True))
                        hx_device = (hx_val.clone().to(device).requires_grad_(True),
                                     hx_val.clone().to(device).add(1).requires_grad_(True))
                    else:
                        hx = hx_val.clone().requires_grad_(True)
                        hx_device = hx_val.clone().to(device).requires_grad_(True)

                    inp = input_val.clone().requires_grad_(True)
                    inp_cu = input_val.clone().to(device).requires_grad_(True)
                    output1, hy1 = rnn(inp, hx)
                    output2, hy2 = rnn_device(inp_cu, hx_device)
                    if is_lstm:
                        torch.autograd.backward(
                            [output1, hy1[0], hy1[1]], [grad_output, grad_hy, grad_hy + 1]
                        )
                        torch.autograd.backward(
                            [output2, hy2[0], hy2[1]],
                            [grad_output.to(device), grad_hy.to(device), (grad_hy + 1).to(device)]
                        )
                    else:
                        torch.autograd.backward([output1, hy1], [grad_output, grad_hy])
                        torch.autograd.backward([output2, hy2], [grad_output.to(device), grad_hy.to(device)])

                    self.assertEqual(output1, output2)
                    self.assertEqual(hy1, hy2)

                    check_rnn_grads(rnn, rnn_device)
                    self.assertEqual(inp.grad, inp_cu.grad)
                    if is_lstm:
                        self.assertEqual(hx[0].grad, hx_device[0].grad)
                        self.assertEqual(hx[1].grad, hx_device[1].grad)
                    else:
                        self.assertEqual(hx.grad, hx_device.grad)

    def _test_rnn_retain_variables(self, device, dtype):
        rnns = [
                nn.GRU(10, 20, num_layers=2).to(device, dtype)]
        for rnn in rnns:
            input = torch.randn(5, 6, 10, device=device, dtype=dtype, requires_grad=True)
            output = rnn(input)
            output[0].sum().backward(retain_graph=True)
            grads = [input.grad.data.clone()] + [p.grad.data.clone() for p in rnn.parameters()]
            for _ in range(4):
                rnn.zero_grad()
                input.grad.data.zero_()
                output[0].sum().backward(retain_graph=True)
                grads2 = [input.grad.data] + [p.grad.data for p in rnn.parameters()]
                self.assertEqual(grads, grads2)

    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.double)
    def test_rnn_retain_variables(self, device, dtype):
        self._test_rnn_retain_variables(device, dtype)

        if self.device_type == 'cuda' and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                self._test_rnn_retain_variables(device, dtype)
    def _test_rnn_mod(self, mod, inp):
        def flatten_out(mod, inp):
            out = mod(inp)
            return tuple([t if isinstance(t, torch.Tensor) else tt for t in out for tt in t])
        gradcheckfunc = partial(flatten_out, mod)
        with torch.backends.cudnn.flags(enabled=False):
            gradcheck(gradcheckfunc, inp, check_batched_grad=False)
            gradgradcheck(gradcheckfunc, inp, check_batched_grad=False)

        if inp.is_cuda and not TEST_WITH_ROCM:
            # Assert that we have good error message around unsupported CuDNN double backward
            # NB: we trigger double backward using .backward() instead of autograd.grad due to
            # https://github.com/pytorch/pytorch/issues/37874
            with torch.backends.cudnn.flags(enabled=True):
                result = gradcheckfunc(inp)
                result[0].sum().backward(create_graph=True)
                grad0 = next(mod.parameters()).grad
                with self.assertRaisesRegex(RuntimeError,
                                            "please disable the CuDNN backend temporarily"):
                    grad0.sum().backward()

                # Here we avoid the backward(create_graph=True) memory leak
                # described in https://github.com/pytorch/pytorch/issues/7343
                for param in mod.parameters():
                    param.grad = None
                inp.grad = None


    @skipMeta  # GRU cell reuses output which was resized
    @dtypes(torch.double)
    def test_GRU_grad_and_gradgrad(self, device, dtype):
        hsize = 4
        inp = torch.rand(1, 3, hsize, device=device, dtype=dtype, requires_grad=True)
        for bias in [True, False]:
            mod = torch.nn.GRU(hsize, hsize, bias=bias).to(device).to(dtype)
            self._test_rnn_mod(mod, inp)

instantiate_device_type_tests(TestNNDeviceType, globals())
instantiate_parametrized_tests(TestNN)

if __name__ == '__main__':
    run_tests()
