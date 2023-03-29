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
     make_fullrank_matrices_with_distinct_singular_values,
     freeze_rng_state, IS_SANDCASTLE)
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


class TestLinalg(TestCase):

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_matrix_rank(self, device, dtype):
        matrix_rank = torch.linalg.matrix_rank

        def run_test(shape0, shape1, batch):
            a = torch.randn(*batch, shape0, shape1, dtype=dtype, device=device)
            rank_a = matrix_rank(a)

            self.assertEqual(rank_a, matrix_rank(a.mH))
            aaH = torch.matmul(a, a.mH)
            rank_aaH = matrix_rank(aaH)
            rank_aaH_hermitian = matrix_rank(aaH, hermitian=True)
            self.assertEqual(rank_aaH, rank_aaH_hermitian)
            aHa = torch.matmul(a.mH, a)
            self.assertEqual(matrix_rank(aHa), matrix_rank(aHa, hermitian=True))

            # check against NumPy
            self.assertEqual(rank_a, np.linalg.matrix_rank(a.cpu().numpy()))
            self.assertEqual(matrix_rank(a, 0.01), np.linalg.matrix_rank(a.cpu().numpy(), 0.01))

            self.assertEqual(rank_aaH, np.linalg.matrix_rank(aaH.cpu().numpy()))
            self.assertEqual(matrix_rank(aaH, 0.01), np.linalg.matrix_rank(aaH.cpu().numpy(), 0.01))

            # hermitian flag for NumPy was added in 1.14.0
            if np.lib.NumpyVersion(np.__version__) >= '1.14.0':
                self.assertEqual(rank_aaH_hermitian,
                                 np.linalg.matrix_rank(aaH.cpu().numpy(), hermitian=True))
                self.assertEqual(matrix_rank(aaH, 0.01, True),
                                 np.linalg.matrix_rank(aaH.cpu().numpy(), 0.01, True))

            # check out= variant
            out = torch.empty(a.shape[:-2], dtype=torch.int64, device=device)
            ans = matrix_rank(a, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(ans, rank_a)

        shapes = (3, 13)
        batches = ((), (0, ), (4, ), (3, 5, ))
        for (shape0, shape1), batch in zip(itertools.product(shapes, reversed(shapes)), batches):
            run_test(shape0, shape1, batch)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_matrix_rank_atol(self, device, dtype):

        def run_test_atol(shape0, shape1, batch):
            a = make_tensor((*batch, shape0, shape1), dtype=dtype, device=device)
            # Check against NumPy output
            # Test float tol, and specific value for each matrix
            tolerances = [float(torch.rand(1)), ]
            # Test different types of tol tensor
            for tol_type in all_types():
                tolerances.append(make_tensor(a.shape[:-2], dtype=tol_type, device=device, low=0))
            # Test broadcasting of tol
            if a.ndim > 2:
                tolerances.append(make_tensor(a.shape[-3], dtype=torch.float32, device=device, low=0))
            for tol in tolerances:
                actual = torch.linalg.matrix_rank(a, atol=tol)
                actual_tol = torch.linalg.matrix_rank(a, tol=tol)
                self.assertEqual(actual, actual_tol)
                numpy_tol = tol if isinstance(tol, float) else tol.cpu().numpy()
                expected = np.linalg.matrix_rank(a.cpu().numpy(), tol=numpy_tol)
                self.assertEqual(actual, expected)

        shapes = (3, 13)
        batches = ((), (0, ), (4, ), (3, 5, ))
        for (shape0, shape1), batch in zip(itertools.product(shapes, reversed(shapes)), batches):
            run_test_atol(shape0, shape1, batch)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float64)
    def test_matrix_rank_atol_rtol(self, device, dtype):
        make_fullrank = make_fullrank_matrices_with_distinct_singular_values
        make_arg = partial(make_fullrank, device=device, dtype=dtype)

        # creates a matrix with singular values rank=n and singular values in range [2/3, 3/2]
        # the singular values are 1 + 1/2, 1 - 1/3, 1 + 1/4, 1 - 1/5, ...
        n = 9
        a = make_arg(n, n)

        # test float and tensor variants
        for tol_value in [0.81, torch.tensor(0.81, device=device)]:
            # using rtol (relative tolerance) takes into account the largest singular value (1.5 in this case)
            result = torch.linalg.matrix_rank(a, rtol=tol_value)
            self.assertEqual(result, 2)  # there are 2 singular values above 1.5*0.81 = 1.215

            # atol is used directly to compare with singular values
            result = torch.linalg.matrix_rank(a, atol=tol_value)
            self.assertEqual(result, 7)  # there are 7 singular values above 0.81

            # when both are specified the maximum tolerance is used
            result = torch.linalg.matrix_rank(a, atol=tol_value, rtol=tol_value)
            self.assertEqual(result, 2)  # there are 2 singular values above max(0.81, 1.5*0.81)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @skipCUDAVersionIn([(11, 6), (11, 7)])  # https://github.com/pytorch/pytorch/issues/75391
    @dtypes(*floating_and_complex_types())
    def test_matrix_rank_empty(self, device, dtype):
        matrix_rank = torch.linalg.matrix_rank

        # NumPy doesn't work for input with no elements
        def run_test(shape0, shape1, batch):
            a = torch.randn(*batch, shape0, shape1, dtype=dtype, device=device)
            rank_a = matrix_rank(a)
            expected = torch.zeros(batch, dtype=torch.int64, device=device)

            self.assertEqual(rank_a, matrix_rank(a.mH))

            aaH = torch.matmul(a, a.mH)
            rank_aaH = matrix_rank(aaH)
            rank_aaH_hermitian = matrix_rank(aaH, hermitian=True)
            self.assertEqual(rank_aaH, rank_aaH_hermitian)

            aHa = torch.matmul(a.mH, a)
            self.assertEqual(matrix_rank(aHa), matrix_rank(aHa, hermitian=True))

            self.assertEqual(rank_a, expected)
            self.assertEqual(matrix_rank(a, 0.01), expected)

            self.assertEqual(rank_aaH, expected)
            self.assertEqual(matrix_rank(aaH, 0.01), expected)

            self.assertEqual(rank_aaH_hermitian, expected)
            self.assertEqual(matrix_rank(aaH, 0.01, True), expected)

        batches = ((), (4, ), (3, 5, ))
        for batch in batches:
            run_test(0, 0, batch)
            run_test(0, 3, batch)
            run_test(3, 0, batch)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_matrix_rank_out_errors_and_warnings(self, device, dtype):
        # dtypes should be safely castable
        a = torch.eye(2, dtype=dtype, device=device)
        out = torch.empty(0, dtype=torch.bool, device=device)
        with self.assertRaisesRegex(RuntimeError, "but got result with dtype Bool"):
            torch.linalg.matrix_rank(a, out=out)

        # device should match
        if torch.cuda.is_available():
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            out = torch.empty(0, dtype=dtype, device=wrong_device)
            with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                torch.linalg.matrix_rank(a, out=out)

        # if out tensor with wrong shape is passed a warning is given
        with warnings.catch_warnings(record=True) as w:
            out = torch.empty(3, dtype=dtype, device=device)
            # Trigger warning
            torch.linalg.matrix_rank(a, out=out)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_matrix_rank_basic(self, device, dtype):
        matrix_rank = torch.linalg.matrix_rank

        a = torch.eye(10, dtype=dtype, device=device)
        self.assertEqual(matrix_rank(a).item(), 10)
        self.assertEqual(matrix_rank(a, hermitian=True).item(), 10)

        a[5, 5] = 0
        self.assertEqual(matrix_rank(a).item(), 9)
        self.assertEqual(matrix_rank(a, hermitian=True).item(), 9)


instantiate_device_type_tests(TestLinalg, globals())

if __name__ == '__main__':
    run_tests()
