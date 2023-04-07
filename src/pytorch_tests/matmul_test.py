# -*- coding: utf-8 -*-
# Owner(s): ["module: linear algebra"]

import torch
import numpy as np

import unittest
import itertools
import warnings
import math
from math import inf, nan, isnan
import random
from random import randrange
from itertools import product
from functools import reduce, partial, wraps

from torch.testing._internal.common_utils import \
    (TestCase, run_tests, TEST_SCIPY, IS_MACOS, IS_WINDOWS, slowTest,
     TEST_WITH_ASAN, TEST_WITH_ROCM, IS_FBCODE, IS_REMOTE_GPU, iter_indices,
     make_fullrank_matrices_with_distinct_singular_values)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, dtypes, has_cusolver,
     onlyCPU, skipCUDAIf, skipCUDAIfNoMagma, skipCPUIfNoLapack, precisionOverride,
     skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfRocm, onlyNativeDeviceTypes, dtypesIfCUDA,
     onlyCUDA, skipCUDAVersionIn, skipMeta, skipCUDAIfNoCusolver, dtypesIfMPS)
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import (
    all_types, all_types_and_complex_and, floating_and_complex_types, integral_types,
    floating_and_complex_types_and, floating_types_and, complex_types,
)
from torch.testing._internal.common_cuda import SM53OrLater, tf32_on_and_off, CUDA11OrLater, CUDA9, _get_magma_version, \
    _get_torch_cuda_version
from torch.distributions.binomial import Binomial

# Protects against includes accidentally setting the default dtype
# NOTE: jit_metaprogramming_utils sets the default dtype to double!
torch.set_default_dtype(torch.float32)
assert torch.get_default_dtype() is torch.float32

if TEST_SCIPY:
    import scipy


class TestLinalg(TestCase):
    def setUp(self):
        super(self.__class__, self).setUp()
        torch.backends.cuda.matmul.allow_tf32 = False

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        super(self.__class__, self).tearDown()

    exact_dtype = True

    def check_single_matmul(self, x, y):

        def assertEqual(answer, expected):
            if x.dtype.is_floating_point or x.dtype.is_complex:
                # Scale the atol with the size of the matrix
                k = max(x.shape[-1], 1)
                self.assertEqual(answer, expected,
                                 msg=f"{x.shape} x {y.shape} = {answer.shape}",
                                 atol=k * 5e-5,
                                 rtol=1e-4)
            else:
                self.assertEqual(answer, expected,
                                 msg=f"{x.shape} x {y.shape} = {answer.shape}")

        # test x @ y
        expected = np.matmul(x.cpu(), y.cpu())
        ans = torch.matmul(x, y)
        self.assertTrue(ans.is_contiguous())
        assertEqual(ans, expected)

        # test out
        out = torch.empty_like(ans)
        ans = torch.matmul(x, y, out=out)
        self.assertIs(ans, out)
        self.assertTrue(ans.is_contiguous())
        assertEqual(ans, expected)

    def gen_sizes_matmul(self, x_dim, y_dim=4, matrix_size=4, batch_size=3):
        """
        Generates sequences of tuples (x, y) of with size(x) = x_dim and
        size(y) <= y_dim that are compatible wrt. matmul
        """
        assert x_dim >= 1
        assert y_dim >= 2
        x = x_dim
        for y in range(1, y_dim + 1):
            for batch, mn in product(product(range(batch_size), repeat=max(x - 2, y - 2, 0)),
                                     product(range(matrix_size), repeat=min(y, 2))):
                if x == 1:
                    size_x = mn[:1]
                    size_y = batch + mn
                    yield size_x, size_y
                else:
                    for k in range(matrix_size):
                        size_x = (k,) + mn[:1]
                        if x > 2:
                            size_x = batch[-(x - 2):] + size_x
                        size_y = mn
                        if y > 2:
                            size_y = batch[-(y - 2):] + size_y
                        yield size_x, size_y

    # Integer matmul just supported on CPU
    @dtypesIfCUDA(torch.float, torch.complex64)
    @dtypes(torch.int64, torch.float, torch.complex64)
    def test_matmul_small_brute_force_1d_Nd(self, device, dtype):
        make_arg = partial(make_tensor, device=device, dtype=dtype)

        for (size_x, size_y), nctg_x, nctg_y in product(self.gen_sizes_matmul(1), (True, False), (True, False)):
            x = make_arg(size_x, noncontiguous=nctg_x)
            y = make_arg(size_y, noncontiguous=nctg_y)
            self.check_single_matmul(x, y)

    # Integer matmul just supported on CPU
    @dtypesIfCUDA(torch.float, torch.complex64)
    @dtypes(torch.int64, torch.float, torch.complex64)
    def test_matmul_small_brute_force_2d_Nd(self, device, dtype):
        make_arg = partial(make_tensor, device=device, dtype=dtype)

        for (size_x, size_y), nctg_x, nctg_y in product(self.gen_sizes_matmul(2), (True, False), (True, False)):
            x = make_arg(size_x, noncontiguous=nctg_x)
            y = make_arg(size_y, noncontiguous=nctg_y)
            self.check_single_matmul(x, y)

    # Integer matmul just supported on CPU
    @dtypesIfCUDA(torch.float, torch.complex64)
    @dtypes(torch.int64, torch.float, torch.complex64)
    def test_matmul_small_brute_force_3d_Nd(self, device, dtype):
        make_arg = partial(make_tensor, device=device, dtype=dtype)

        for (size_x, size_y), nctg_x, nctg_y in product(self.gen_sizes_matmul(3), (True, False), (True, False)):
            x = make_arg(size_x, noncontiguous=nctg_x)
            y = make_arg(size_y, noncontiguous=nctg_y)
            self.check_single_matmul(x, y)


instantiate_device_type_tests(TestLinalg, globals())

if __name__ == '__main__':
    run_tests()
