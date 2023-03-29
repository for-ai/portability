# Owner(s): ["module: cuda"]

from itertools import repeat, chain, product
from typing import NamedTuple
import collections
import contextlib
from copy import deepcopy
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
from random import randint

import torch
import torch.cuda
import torch.cuda.comm as comm
from torch.nn.parallel import scatter_gather
from torch.utils.checkpoint import checkpoint_sequential
from torch._six import inf, nan
from torch.testing._internal.common_utils import TestCase, freeze_rng_state, run_tests, \
    NO_MULTIPROCESSING_SPAWN, skipIfRocm, load_tests, IS_REMOTE_GPU, IS_SANDCASTLE, IS_WINDOWS, \
    slowTest, skipCUDANonDefaultStreamIf, skipCUDAMemoryLeakCheckIf, TEST_WITH_ROCM, TEST_NUMPY, \
    get_cycles_per_ms, parametrize, instantiate_parametrized_tests, subtest
from torch.testing._internal.autocast_test_lists import AutocastTestLists

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

# We cannot import TEST_CUDA and TEST_MULTIGPU from torch.testing._internal.common_cuda here,
# because if we do that, the TEST_CUDNN line from torch.testing._internal.common_cuda will be executed
# multiple times as well during the execution of this test suite, and it will
# cause CUDA OOM error on Windows.
TEST_CUDA = torch.cuda.is_available()
TEST_MULTIGPU = TEST_CUDA and torch.cuda.device_count() >= 2

if not TEST_CUDA:
    print('CUDA not available, skipping tests', file=sys.stderr)
    TestCase = object  # noqa: F811

TEST_CUDAMALLOCASYNC = TEST_CUDA and (torch.cuda.get_allocator_backend() == "cudaMallocAsync")
TEST_LARGE_TENSOR = TEST_CUDA
TEST_MEDIUM_TENSOR = TEST_CUDA
TEST_CUDNN = TEST_CUDA
TEST_BF16 = False
if TEST_CUDA:
    torch.ones(1).cuda()  # initialize cuda context
    TEST_CUDNN = TEST_CUDA and (TEST_WITH_ROCM or
                                torch.backends.cudnn.is_acceptable(torch.tensor(1., device=torch.device('cuda:0'))))
    TEST_LARGE_TENSOR = torch.cuda.get_device_properties(0).total_memory >= 12e9
    TEST_MEDIUM_TENSOR = torch.cuda.get_device_properties(0).total_memory >= 6e9
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
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True
    FIFTY_MIL_CYCLES = 50000000

    def setUp(self):
        super(TestCuda, self).setUp()
        self.autocast_lists = AutocastTestLists(torch.device('cuda:0'))

    def tearDown(self):
        del self.autocast_lists
        super(TestCuda, self).tearDown()

    def _check_memory_stat_consistency(self):
        snapshot = torch.cuda.memory_snapshot()

        expected_each_device = collections.defaultdict(lambda: collections.defaultdict(int))

        for segment in snapshot:
            expected = expected_each_device[segment["device"]]
            pool_str = segment["segment_type"] + "_pool"

            expected["segment.all.current"] += 1
            expected["segment." + pool_str + ".current"] += 1

            expected["allocated_bytes.all.current"] += segment["allocated_size"]
            expected["allocated_bytes." + pool_str + ".current"] += segment["allocated_size"]

            expected["reserved_bytes.all.current"] += segment["total_size"]
            expected["reserved_bytes." + pool_str + ".current"] += segment["total_size"]

            expected["active_bytes.all.current"] += segment["active_size"]
            expected["active_bytes." + pool_str + ".current"] += segment["active_size"]

            is_split = len(segment["blocks"]) > 1
            for block in segment["blocks"]:
                if block["state"] == "active_allocated":
                    expected["allocation.all.current"] += 1
                    expected["allocation." + pool_str + ".current"] += 1

                if block["state"].startswith("active_"):
                    expected["active.all.current"] += 1
                    expected["active." + pool_str + ".current"] += 1

                if block["state"] == "inactive" and is_split:
                    expected["inactive_split.all.current"] += 1
                    expected["inactive_split." + pool_str + ".current"] += 1
                    expected["inactive_split_bytes.all.current"] += block["size"]
                    expected["inactive_split_bytes." + pool_str + ".current"] += block["size"]

        for device, expected in expected_each_device.items():
            stats = torch.cuda.memory_stats(device)
            for k, v in expected.items():
                self.assertEqual(v, stats[k])

    @staticmethod
    def _test_memory_stats_generator(self, device=None, N=35):
        if device is None:
            device = torch.cuda.current_device()

        m0 = torch.cuda.memory_allocated(device)
        last_m_arr = [torch.cuda.memory_allocated(device)]
        max_m_arr = [torch.cuda.max_memory_allocated(device)]
        last_r_arr = [torch.cuda.memory_reserved(device)]
        max_r_arr = [torch.cuda.max_memory_reserved(device)]

        def alloc(*size):
            with torch.cuda.device(device):
                # NOTE: do **not** use methods that can have additional
                #       memory overhead, e.g., inplace random sampling methods.
                #       they can leave some memory occupied even after being
                #       deallocated, e.g., initialized RNG state, causing some
                #       memory checks below to fail.
                return torch.cuda.FloatTensor(*size)

        def assert_change(comp=1, empty_cache=False, reset_peak=False):
            # comp > 0: increased
            # comp = 0: equal
            # comp < 0: decreased
            new_m = torch.cuda.memory_allocated(device)
            new_max_m = torch.cuda.max_memory_allocated(device)
            if comp > 0:
                self.assertGreater(new_m, last_m_arr[0])
            elif comp < 0:
                self.assertLess(new_m, last_m_arr[0])
            else:
                self.assertEqual(new_m, last_m_arr[0])
            self.assertLessEqual(new_m, new_max_m)
            self.assertGreaterEqual(new_max_m, max_m_arr[0])
            last_m_arr[0] = new_m
            max_m_arr[0] = new_max_m

            new_r = torch.cuda.memory_reserved(device)
            new_max_r = torch.cuda.max_memory_reserved(device)
            # emptying cache may happen (due to allocation or empty_cache), so
            # we can't assert new_c >= last_c
            self.assertLessEqual(new_r, new_max_r)
            self.assertGreaterEqual(new_max_r, max_r_arr[0])
            last_r_arr[0] = new_r
            max_r_arr[0] = new_max_r

            if empty_cache:
                torch.cuda.empty_cache()
                new_r = torch.cuda.memory_reserved(device)
                new_max_r = torch.cuda.max_memory_reserved(device)
                self.assertLessEqual(new_r, last_r_arr[0])
                self.assertLessEqual(new_r, new_max_r)
                self.assertEqual(new_max_r, max_r_arr[0])
                last_r_arr[0] = new_r

            if reset_peak:
                torch.cuda.reset_peak_memory_stats(device)
                self.assertEqual(torch.cuda.memory_allocated(device), last_m_arr[0])
                self.assertEqual(torch.cuda.max_memory_allocated(device), last_m_arr[0])
                max_m_arr[0] = last_m_arr[0]
                self.assertEqual(torch.cuda.memory_reserved(device), last_r_arr[0])
                self.assertEqual(torch.cuda.max_memory_reserved(device), last_r_arr[0])
                max_r_arr[0] = last_r_arr[0]

        assert_change(0)
        assert_change(0, reset_peak=True)
        assert_change(0, empty_cache=True)
        assert_change(0, reset_peak=True)
        assert_change(0)
        yield

        tensors1 = [alloc(1), alloc(10, 20), alloc(200, 300, 2000)]
        m1 = torch.cuda.memory_allocated(device)
        assert_change(1)
        yield

        tensors2 = []

        for i in range(1, int(N / 2) + 1):
            # small ones
            tensors2.append(alloc(i, i * 4))
            assert_change(1)
            yield

        for i in range(5, int(N / 2) + 5):
            # large ones
            tensors2.append(alloc(i, i * 7, i * 9, i * 11))
            assert_change(1, reset_peak=(i % 2 == 0))
            yield

        tensors2.append(alloc(0, 0, 0))
        assert_change(0)
        yield

        permute = []
        for i in torch.randperm(len(tensors2)):
            permute.append(tensors2[i])
            assert_change(0)
            yield

        del tensors2
        assert_change(0)
        yield
        tensors2 = permute
        assert_change(0)
        yield
        del permute
        assert_change(0, reset_peak=True)
        yield

        for i in range(int(N / 2)):
            x = tensors2[i].numel()
            del tensors2[i]
            assert_change(-x)  # in case that tensors2[i] is empty
            yield

        for i in range(2, int(2 * N / 3) + 2):
            tensors2.append(alloc(i, i * 3, i * 8))
            assert_change(1)
            yield

        del tensors2
        assert_change(-1, reset_peak=True)
        assert_change(0)
        self.assertEqual(torch.cuda.memory_allocated(device), m1)
        yield True

        del tensors1
        assert_change(-1, reset_peak=True)
        self.assertEqual(torch.cuda.memory_allocated(device), m0)

        # test empty_cache and reset_peak
        assert_change(0, empty_cache=True)
        assert_change(0, reset_peak=True)



    @unittest.skipIf(TEST_CUDAMALLOCASYNC, "temporarily disabled")
    def test_memory_stats(self):
        gc.collect()
        torch.cuda.empty_cache()
        for _ in self._test_memory_stats_generator(self):
            self._check_memory_stat_consistency()

    @unittest.skipIf(TEST_CUDAMALLOCASYNC, "temporarily disabled")
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_memory_stats_multigpu(self):
        # advance a generator with a end flag
        def advance(gen, end):
            if not end:
                try:
                    next(gen)
                except StopIteration:
                    end = True
            return end

        # interlace
        torch.cuda.empty_cache()
        gen0 = self._test_memory_stats_generator(self, device='cuda:0', N=35)
        gen1 = self._test_memory_stats_generator(self, device=torch.device('cuda:1'), N=35)
        end0 = end1 = False
        while not (end0 and end1):
            end0 = advance(gen0, end0)
            end1 = advance(gen1, end1)

        # semi-random order
        torch.cuda.empty_cache()
        gen0 = self._test_memory_stats_generator(self, device=0, N=35)
        gen1 = self._test_memory_stats_generator(self, device=torch.device('cuda:1'), N=35)
        end0 = end1 = False

        while not (end0 and end1):
            end0 = advance(gen0, end0)
            if not end0:
                gen1_max_times = torch.LongTensor(1).random_(0, 3)[0]
            else:
                gen1_max_times = inf
            t = 0
            while t < gen1_max_times and not end1:
                end1 = advance(gen1, end1)
                t += 1

    
instantiate_parametrized_tests(TestCuda)

if __name__ == '__main__':
    run_tests()
