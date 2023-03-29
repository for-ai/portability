# Owner(s): ["module: cuda"]

from itertools import repeat, chain, product
from typing import NamedTuple
import collections
import contextlib
import ctypes
import gc
import io
import os
import pickle
import queue
import sys
import tempfile
import threading
import unittest

import torch
import torch.cuda
import torch.cuda.comm as comm
from torch.nn.parallel import scatter_gather
from torch.utils.checkpoint import checkpoint_sequential
from torch._six import inf, nan
from torch.testing._internal.common_methods_invocations import tri_tests_args, tri_large_tests_args, \
    _compare_trilu_indices, _compare_large_trilu_indices
from torch.testing._internal.common_utils import TestCase, freeze_rng_state, run_tests, \
    NO_MULTIPROCESSING_SPAWN, skipIfRocm, load_tests, IS_REMOTE_GPU, IS_SANDCASTLE, IS_WINDOWS, \
    slowTest, skipCUDANonDefaultStreamIf, skipCUDAMemoryLeakCheckIf, TEST_WITH_ROCM, TEST_NUMPY, \
    get_cycles_per_ms
from torch.testing._internal.autocast_test_lists import AutocastTestLists
from ..utils.timer_wrapper import pytorch_op_timer
from ..utils.pytorch_device_decorators import onlyNativeDeviceTypes, onlyAcceleratedDeviceTypes, instantiate_device_type_tests

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

# We cannot import TEST_CUDA and TEST_MULTIGPU from torch.testing._internal.common_cuda here,
# because if we do that, the TEST_CUDNN line from torch.testing._internal.common_cuda will be executed
# multiple times as well during the execution of this test suite, and it will
# cause CUDA OOM error on Windows.
TEST_CUDA = torch.cuda.is_available()
TEST_MULTIGPU = TEST_CUDA and torch.cuda.device_count() >= 2

TEST_LARGE_TENSOR = TEST_CUDA
TEST_MEDIUM_TENSOR = TEST_CUDA
TEST_CUDNN = TEST_CUDA
TEST_BF16 = False
if TEST_CUDA:
    torch.ones(1).cuda()  # initialize cuda context
    TEST_CUDNN = TEST_CUDA and (TEST_WITH_ROCM or
                                torch.backends.cudnn.is_acceptable(torch.tensor(1., device=torch.device('cuda:0'))))
    TEST_LARGE_TENSOR = torch.cuda.get_device_properties(
        0).total_memory >= 12e9
    TEST_MEDIUM_TENSOR = torch.cuda.get_device_properties(
        0).total_memory >= 6e9
    TEST_BF16 = torch.cuda.is_bf16_supported()


def make_sparse_tensor(t, n, *sizes):
    assert t.is_sparse
    tensor = t()
    i = tensor._indices()
    i = i.new(len(sizes), n).copy_(
        torch.cat([torch.LongTensor(1, n).random_(s) for s in sizes], 0))
    v = tensor._values()
    v = v.new(n).copy_(torch.randn(n))
    return t(i, v, torch.Size(sizes)).coalesce()


_cycles_per_ms = None


class TestCuda(TestCase):
    FIFTY_MIL_CYCLES = 50000000

    def _test_copy_sync_current_stream(self, x, y):
        x_plus_one = x + 1
        s0 = torch.cuda.Stream(device=x.device)
        s1 = torch.cuda.Stream(device=y.device)
        s2 = torch.cuda.Stream(device=x.device)
        s3 = torch.cuda.Stream(device=y.device)

        # same dst stream different src streams
        with torch.cuda.stream(s0):
            torch.cuda._sleep(TestCuda.FIFTY_MIL_CYCLES)
            with torch.cuda.stream(s1):
                y.copy_(x_plus_one)

        with torch.cuda.stream(s2), torch.cuda.stream(s1):
            y.copy_(x)

        with pytorch_op_timer():
            s1.synchronize()
        # The copy() is synchronized on the current streams of both src and dst.
        # In the above test, the _sleep() op on s0 will not block the copy() on
        # s2, but both copies are synchronized on s1 in the dst device. Hence,
        # x is copied to y after x_plus_one is copied to y. If x and y are on
        # the same device, both copy() ops are synchronized on s1.
        self.assertEqual(y, x)

        # same src stream different dst streams
        with torch.cuda.stream(s1):
            torch.cuda._sleep(TestCuda.FIFTY_MIL_CYCLES)
            with torch.cuda.stream(s0):
                y.copy_(x_plus_one)

        with torch.cuda.stream(s3), torch.cuda.stream(s0):
            y.copy_(x)

        with pytorch_op_timer():
            s0.synchronize()
        # Similarly, both copy() ops are synchronized on s0.
        self.assertEqual(y, x)

    # @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    # @onlyAcceleratedDeviceTypes
    def test_copy_streams(self):
        d0 = torch.device('cuda:0')
        x0 = torch.zeros(5, 5, device=d0)

        d1 = torch.device('cuda:0')
        x1 = torch.zeros(5, 5, device=d1)
        self._test_copy_sync_current_stream(x0, x1)

        x2 = torch.zeros(5, 5, device=d0)
        self._test_copy_sync_current_stream(x0, x2)

# instantiate_device_type_tests(TestCuda, globals())
if __name__ == '__main__':
    run_tests()
