# Owner(s): ["module: autograd"]

import contextlib
import gc
import io
import math
import os
import random
import sys
import tempfile
import threading
import time
import unittest
import uuid
import warnings
import operator
import subprocess
from copy import deepcopy
from collections import OrderedDict
from itertools import product
from operator import mul
from functools import reduce, partial
import torch

from torch import nn
from torch._six import inf, nan
from torch.autograd.function import once_differentiable
from torch.autograd.profiler import (profile, record_function, emit_nvtx)
from torch.autograd.profiler_util import (
    _format_time, EventList, FunctionEvent, FunctionEventAvg)
from torch.utils.checkpoint import checkpoint
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    TestCase, run_tests, skipIfNoLapack, slowTest, IS_WINDOWS, IS_MACOS,
    disable_gc, gradcheck, gradgradcheck, parametrize,
    instantiate_parametrized_tests, skipIfMps, set_warn_always_context)
from torch.autograd import Variable, Function, detect_anomaly, kineto_available
from torch.autograd.function import InplaceFunction
import torch.autograd.forward_ad as fwAD
from torch.testing._internal.common_methods_invocations import mask_not_all_zeros
from torch.testing._internal.common_device_type import (instantiate_device_type_tests,
                                                        onlyCPU, onlyCUDA, dtypes, dtypesIfCUDA,
                                                        deviceCountAtLeast, skipMeta, dtypesIfMPS)
from torch.testing._internal.common_dtype import floating_types_and
import weakref

import pickle

from ..utils.pytorch_device_decorators import onlyNativeDeviceTypes, onlyAcceleratedDeviceTypes, instantiate_device_type_tests, onlyGPU, onlyTPU, skipCUDAIfNoCudnn
from ..utils.timer_wrapper import pytorch_op_timer


def graph_desc(fn):
    if fn is None:
        return 'None'
    result = type(fn).__name__ + '('
    next_functions = fn.next_functions
    for next_fn, _ in next_functions:
        result += graph_desc(next_fn)
        result += ', '
    if next_functions:
        result = result[:-2]
    return result + ')'


class TestAutograd(TestCase):

    def test_indexing(self, device):
        x = torch.arange(1., 17, device=device).view(4, 4)
        with pytorch_op_timer():
            y = Variable(x, requires_grad=True).to(device)

        def compare(x, y, idx, indexed_tensor, indexed_var):
            indexed_var_t = indexed_var.data
            if not isinstance(indexed_tensor, torch.Tensor):
                indexed_var_t = indexed_var_t[0]
            self.assertEqual(indexed_tensor, indexed_var_t)

            indexed_var.sum().backward()
            expected_grad = torch.empty(x.size()).fill_(0)
            expected_grad[idx] = 1
            self.assertEqual(y.grad, expected_grad)

        def check_index(x, y, idx):
            if y.grad is not None:
                with torch.no_grad():
                    y.grad.zero_()
            indexed_tensor = x[idx]
            indexed_var = y[idx]
            compare(x, y, idx, indexed_tensor, indexed_var)

        check_index(x, y, 1)
        check_index(x, y, (1, 1))
        check_index(x, y, slice(1, None))
        check_index(x, y, slice(None, 2))
        check_index(x, y, (slice(None, 2), 2))
        check_index(x, y, (slice(1, 2), 2))
        check_index(x, y, (1, slice(2, None)))
        check_index(x, y, (slice(None, None), slice(2, None)))
        check_index(x, y, torch.LongTensor([0, 2]))
        check_index(x, y, torch.rand(4, 4).bernoulli().bool())
        check_index(x, y, (Ellipsis, slice(2, None)))
        check_index(x, y, ([0], [0]))
        check_index(x, y, ([1, 2, 3], [0]))
        check_index(x, y, ([1, 2], [2, 1]))
        check_index(x, y, ([[1, 2], [3, 0]], [[0, 1], [2, 3]]))
        check_index(x, y, ([slice(None), [2, 3]]))
        check_index(x, y, ([[2, 3], slice(None)]))

        # advanced indexing, with less dim, or ellipsis
        check_index(x, y, ([0]))
        check_index(x, y, ([0], ))

        x = torch.arange(1., 49).view(4, 3, 4)
        with pytorch_op_timer():
            y = Variable(x, requires_grad=True)

        check_index(x, y, (slice(None), [0], [0]))
        check_index(x, y, ([0], [0], slice(None)))
        check_index(x, y, (slice(None), [0, 1, 2], [0]))
        check_index(x, y, ([0, 1, 2], [0], slice(None)))
        check_index(x, y, (slice(None), [1, 2], [2, 1]))
        check_index(x, y, ([1, 2], [2, 1], slice(None)))
        check_index(x, y, (slice(None), [[1, 2], [2, 0]], [[0, 1], [2, 3]]))
        check_index(x, y, ([[1, 2], [3, 0]], [[0, 1], [2, 2]], slice(None)))
        check_index(x, y, (slice(None), slice(None), [2, 1]))
        check_index(x, y, (slice(None), [2, 1], slice(None)))
        check_index(x, y, ([2, 1], slice(None), slice(None)))

        # advanced indexing, with less dim, or ellipsis
        check_index(x, y, ([0], ))
        check_index(x, y, ([0], slice(None)))
        check_index(x, y, ([0], Ellipsis))
        check_index(x, y, ([1, 2], [0, 1]))
        check_index(x, y, ([1, 2], [0, 1], Ellipsis))
        check_index(x, y, (Ellipsis, [1, 2], [0, 1]))

        # advanced indexing, with a tensor wrapped in a variable
        z = torch.LongTensor([0, 1]).to(device)
        with pytorch_op_timer():
            zv = Variable(z, requires_grad=False).to(device)
        seq = [z, Ellipsis]
        seqv = [zv, Ellipsis]

        if y.grad is not None:
            with torch.no_grad():
                y.grad.zero_()
        indexed_tensor = x[seq]
        indexed_var = y[seqv]
        compare(x, y, seq, indexed_tensor, indexed_var)

    def test_indexing_duplicates(self, device):
        x = torch.arange(1., 17, device=device).view(4, 4)
        with pytorch_op_timer():
            y = Variable(x, requires_grad=True)

        idx = torch.LongTensor([1, 1, 3, 2, 1, 2])
        y[idx].sum().backward()
        expected_grad = torch.zeros(4, 4, device=device)
        for i in idx:
            expected_grad[i] += 1
        self.assertEqual(y.grad, expected_grad)

        # with advanced indexing
        x = torch.arange(1., 17, device=device).view(4, 4)
        with pytorch_op_timer():
            y = Variable(x, requires_grad=True).to(device)

        idx = [[1, 1, 3, 2, 1, 2], [0]]
        y[idx].sum().backward()
        expected_grad = torch.zeros(4, 4, device=device)
        for i in idx[0]:
            for j in idx[1]:
                expected_grad[i][j] += 1

        self.assertEqual(y.grad, expected_grad)

        x = torch.arange(1., 17, device=device).view(4, 4)
        with pytorch_op_timer():
            y = Variable(x, requires_grad=True).to(device)
        idx = [[[1, 2], [0, 0]], [[0, 1], [1, 1]]]
        y[idx].sum().backward()
        expected_grad = torch.tensor([[0., 2., 0., 0.],
                                      [1., 0., 0., 0.],
                                      [0., 1., 0., 0.],
                                      [0., 0., 0., 0.]], device=device)
        self.assertEqual(y.grad, expected_grad)

        x = torch.arange(1., 65, device=device).view(4, 4, 4)
        with pytorch_op_timer():
            y = Variable(x, requires_grad=True).to(device)

        idx = [[1, 1, 1], slice(None), slice(None)]
        y[idx].sum().backward()
        expected_grad = torch.empty(4, 4, 4).zero_()
        expected_grad[1].fill_(3)
        self.assertEqual(y.grad, expected_grad)

    def test_inplace(self, device):
        x = torch.ones(5, 5, requires_grad=True, device=device)
        with pytorch_op_timer():
            y = Variable(torch.ones(5, 5) * 4, requires_grad=True).to(device)

        z = x * y
        q = z + y
        w = z * y
        z.add_(2)
        # Add doesn't need it's inputs to do backward, so it shouldn't raise
        q.backward(torch.ones(5, 5, device=device), retain_graph=True)
        # Mul saves both inputs in forward, so it should raise
        self.assertRaises(RuntimeError, lambda: w.backward(torch.ones(5, 5)))

        z = x * y
        q = z * y
        r = z + y
        w = z.add_(y)
        # w is a the last expression, so this should succeed
        w.backward(torch.ones(5, 5, device=device), retain_graph=True)
        # r doesn't use the modified value in backward, so it should succeed
        r.backward(torch.ones(5, 5, device=device), retain_graph=True)
        # q uses dirty z, so it should raise
        self.assertRaises(RuntimeError, lambda: q.backward(torch.ones(5, 5)))

        with torch.no_grad():
            x.grad.zero_()
        m = x / 2
        z = m + y / 8
        q = z * y
        r = z + y
        prev_version = z._version
        w = z.exp_()
        self.assertNotEqual(z._version, prev_version)
        r.backward(torch.ones(5, 5, device=device), retain_graph=True)
        self.assertEqual(x.grad, torch.ones(5, 5, device=device) / 2)
        w.backward(torch.ones(5, 5, device=device), retain_graph=True)
        self.assertEqual(x.grad, torch.empty(
            5, 5, device=device).fill_((1 + math.e) / 2))
        self.assertRaises(RuntimeError, lambda: q.backward(
            torch.ones(5, 5, device=device)))

        leaf = torch.ones(5, 5, requires_grad=True, device=device)
        x = leaf.clone()
        x.add_(10)
        self.assertEqual(x, torch.ones(5, 5, device=device) * 11)
        # x should be still usable
        y = x + 2
        y.backward(torch.ones(5, 5, device=device))
        self.assertEqual(leaf.grad, torch.ones(5, 5, device=device))
        z = x * y
        x.add_(2)
        self.assertRaises(RuntimeError, lambda: z.backward(
            torch.ones(5, 5, device=device)))

    def _test_type_conversion_backward(self, t):
        with pytorch_op_timer():
            fvar = Variable(t(torch.randn(5, 5).float()), requires_grad=True)
        fvar.double().sum().backward()
        self.assertEqual(fvar.grad, torch.ones_like(fvar))
        self.assertEqual(type(fvar.grad), type(fvar))
        with pytorch_op_timer():
            dvar = Variable(t(torch.randn(5, 5).double()), requires_grad=True)
        dvar.float().sum().backward()
        self.assertEqual(dvar.grad, torch.ones_like(dvar))
        self.assertEqual(type(dvar.grad), type(dvar))

    # @onlyAcceleratedDeviceTypes
    # def test_type_conversions(self, device):
    #     x = torch.randn(5, 5)
    #     self.assertIsInstance(x.float(), torch.FloatTensor)
    #     self.assertIsInstance(x.int(), torch.IntTensor)
    #     # if torch.cuda.is_available():
    #     self.assertIsInstance(x.float().to(device), torch.cuda.FloatTensor)
    #     self.assertIsInstance(x.int().to(device), torch.cuda.IntTensor)
    #     self.assertIsInstance(x.int().to(device).cpu(), torch.IntTensor)
    #         # if torch.cuda.device_count() >= 2:
    #         #     x2 = x.float().to(device)
    #         #     self.assertIsInstance(x2, torch.cuda.FloatTensor)
    #         #     self.assertIs(x2.get_device(), 1)
    #         #     x2 = x.float().cuda()
    #         #     self.assertIsInstance(x2, torch.cuda.FloatTensor)
    #         #     self.assertIs(x2.get_device(), 0)
    #         #     x2 = x2.cuda(1)
    #         #     self.assertIsInstance(x2, torch.cuda.FloatTensor)
    #         #     self.assertIs(x2.get_device(), 1)
    #         #     with pytorch_op_timer():
    #         #         y = Variable(torch.randn(5).cuda(1), requires_grad=True)
    #         #     y.cpu().sum().backward()
    #         #     self.assertIs(y.grad.get_device(), 1)
    #         #     self.assertIs(y.long().get_device(), 1)

        for t in [torch.DoubleTensor, torch.FloatTensor, torch.IntTensor, torch.ByteTensor]:
            for y_var in (True, False):
                y = torch.randint(5, (5, 5), dtype=t.dtype)
                with pytorch_op_timer():
                    y = Variable(y) if y_var else y
                self.assertIsInstance(x.type(t), t)
                self.assertIsInstance(x.type_as(y), t)
                # TODO: t.dtype should work
                t_dtype = t().dtype
                self.assertIsInstance(x.type(t_dtype), t)
                self.assertIs(t_dtype, x.type(t_dtype).dtype)
                self.assertEqual(y.data_ptr(), y.type(t).data_ptr())
                # if torch.cuda.is_available():
                for x_cuda in (True, False):
                    for y_cuda in (True, False):
                        x_c = x.to(device) if x_cuda else x
                        y_c = y.to(device) if y_cuda else y
                        _, y_type = y_c.type().rsplit('.', 1)
                        y_typestr = (
                            'torch.cuda.' if y_cuda else 'torch.') + y_type
                        self.assertEqual(
                            y_c.type(), x_c.type(y_typestr).type())
                        self.assertIs(y_c.dtype, x_c.type(y_c.dtype).dtype)
                        self.assertEqual(y_c.data_ptr(), y_c.to(
                            device).data_ptr() if y_cuda else y_c.data_ptr())

        self._test_type_conversion_backward(lambda x: x)
        # if torch.cuda.is_available():
        self._test_type_conversion_backward(lambda x: x.to(device))
        # if torch.cuda.device_count() >= 2:
        #     # one of these has to be the non-default device
        #     self._test_type_conversion_backward(lambda x: x.cuda(0))
        #     self._test_type_conversion_backward(lambda x: x.cuda(1))

    def test_simple_reentrant(self, device):
        y_data = torch.randn(2, 2, device=device)

        class Reenter(Function):
            @staticmethod
            def forward(ctx, x):
                with torch.enable_grad():
                    with pytorch_op_timer():
                        ctx.x = Variable(x, requires_grad=True).to(device)
                    with pytorch_op_timer():
                        ctx.y = Variable(y_data, requires_grad=True).to(device)
                    ctx.output_var = ctx.x * ctx.y
                return ctx.output_var.detach()

            @staticmethod
            def backward(ctx, grad_output):
                with torch.enable_grad():
                    ctx.output_var.sum().backward()
                return ctx.x.grad * grad_output

        # Reentrant starts on CPU thread, finishs on GPU thread
        x = torch.randn(2, 2, requires_grad=True, device=device)
        out = Reenter.apply(x)
        out.sum().backward()
        self.assertEqual(x.grad, y_data)

    @skipIfMps  # the test doesn't work on MPS as double types are not supported
    def test_pyscalar_conversions(self, device):
        def _test_pyscalar_conversions(t, integral_conv):
            # integral -> integral
            l = t(torch.zeros(1, 1, 1, dtype=torch.long, device=device))
            pyscalar = -12345
            l[0] = pyscalar
            self.assertEqual(integral_conv(l), pyscalar)

            # floating point -> floating point
            with pytorch_op_timer():
                f = Variable(
                    t(torch.randn(1, 1, dtype=torch.double))).to(device)
            pyscalar = -12345.1
            f[0] = pyscalar
            self.assertEqual(float(f), pyscalar)
            f[0] = nan
            self.assertTrue(math.isnan(float(f)))
            f[0] = inf
            self.assertEqual(float(f), inf)
            f[0] = -inf
            self.assertEqual(float(f), -inf)

            # integral -> floating point
            # check we can convert something that loses precision
            pyscalar = 1234567890123456789
            self.assertNotEqual(pyscalar, integral_conv(float(pyscalar)))
            l[0] = pyscalar
            self.assertEqual(float(l), float(pyscalar))

            # floating point -> integral
            f[0] = nan
            self.assertRaises(ValueError, lambda: integral_conv(f[0]))
            f[0] = inf
            self.assertRaises(OverflowError, lambda: integral_conv(f[0]))
            f[0] = -inf
            self.assertRaises(OverflowError, lambda: integral_conv(f[0]))
            f[0] = sys.float_info.max
            self.assertEqual(integral_conv(f), sys.float_info.max)

            # bool, nonzero
            def test_nonzero(tensor, value, expected):
                tensor[0] = value
                self.assertEqual(expected, bool(tensor))
                self.assertEqual(expected, True if tensor else False)

            test_nonzero(l, 0, False)
            test_nonzero(l, -2, True)
            test_nonzero(f, 0.0, False)
            test_nonzero(f, sys.float_info.min, True)
            test_nonzero(f, nan, bool(nan))
            test_nonzero(f, inf, bool(inf))
            test_nonzero(f, -inf, bool(-inf))

        _test_pyscalar_conversions(lambda x: x.to(device), lambda x: int(x))

    def test_reentrant_priority(self, device):
        order = []

        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, x):
                order.append("MyFunction")
                return x

        class Reentrant(Function):
            @staticmethod
            def forward(ctx, x):
                with torch.enable_grad():
                    with pytorch_op_timer():
                        ctx.x = Variable(
                            x.detach(), requires_grad=True)  # .to(device)
                    ctx.x = ctx.x - 1
                return ctx.x.detach()

            @staticmethod
            def backward(ctx, x):
                order.append("Reentrant")
                if ctx.x < 0:
                    return x
                with torch.enable_grad():
                    Reentrant.apply(ctx.x).backward()
                return x

        # ['Reentrant', 'MyFunction', 'Reentrant', 'Reentrant', 'Reentrant', 'Reentrant', 'Reentrant', 'Reentrant', 'Reentrant', 'Reentrant', 'Reentrant']
        a = MyFunction.apply(torch.tensor(
            6.0, requires_grad=True, device=device))
        b = Reentrant.apply(torch.tensor(
            9.0, requires_grad=True, device=device))
        v = a * b
        v.backward()
        # The tasks for the Reentrant and MyFunction backward() will be added
        # to the queue in the autograd engine at the same time. The backward
        # for Reentrant will be executed first, which will then add other
        # backward tasks to the queue. We want to ensure all the reentrant tasks
        # are prioritized over the MyFunction backward task regardless of their
        # sequence numbers
        self.assertEqual(len(order), 11)
        self.assertEqual(order.count("Reentrant"), 10)
        self.assertEqual(order[-1], "MyFunction")

    # @unittest.skipIf(IS_MACOS, "Fails with SIGBUS on macOS; https://github.com/pytorch/pytorch/issues/25941")
    # def test_deep_reentrant(self, device):

    #     class DeepReentrant(Function):
    #         @staticmethod
    #         def forward(ctx, x):
    #             with torch.enable_grad():
    #                 with pytorch_op_timer():
    #                     ctx.x = Variable(x.detach(), requires_grad=True).to(device)
    #                 ctx.x = ctx.x - 1
    #             return ctx.x.detach()

    #         @staticmethod
    #         def backward(ctx, x):
    #             if ctx.x < 0:
    #                 return x
    #             with torch.enable_grad():
    #                 DeepReentrant.apply(ctx.x).sum().backward()
    #             return x

    #     # Test stack overflow escape mechanism
    #     v = torch.tensor(2000.0, requires_grad=True)
    #     # This will cause stack overflow if reentrant calls are handled
    #     # in the same thread recursively
    #     DeepReentrant.apply(v).sum().backward()

    #     # Test stack overflow escape mechanism multiple times
    #     # to ensure reusing workers in the pool works fine
    #     v2 = torch.tensor(200.0, requires_grad=True, device=device)
    #     DeepReentrant.apply(v2).sum().backward()


instantiate_device_type_tests(TestAutograd, globals())

if __name__ == '__main__':
    run_tests()
