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

if TEST_SCIPY:
    import scipy

def setLinalgBackendsToDefaultFinally(fn):
    @wraps(fn)
    def _fn(*args, **kwargs):
        try:
            fn(*args, **kwargs)
        finally:
            # Set linalg backend back to default to make sure potential failures in one test
            #   doesn't affect other linalg tests
            torch.backends.cuda.preferred_linalg_library('default')
    return _fn

class TestLinalg(TestCase):
    def setUp(self):
        super(self.__class__, self).setUp()
        torch.backends.cuda.matmul.allow_tf32 = False

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        super(self.__class__, self).tearDown()

    exact_dtype = True

    def cholesky_solve_test_helper(self, A_dims, b_dims, upper, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        b = torch.randn(*b_dims, dtype=dtype, device=device)
        A = random_hermitian_pd_matrix(*A_dims, dtype=dtype, device=device)
        L = torch.cholesky(A, upper=upper)
        return b, A, L

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    def test_cholesky_solve(self, device, dtype):
        for (k, n), upper in itertools.product(zip([2, 3, 5], [3, 5, 7]), [True, False]):
            b, A, L = self.cholesky_solve_test_helper((n,), (n, k), upper, device, dtype)
            x = torch.cholesky_solve(b, L, upper=upper)
            self.assertEqual(b, np.matmul(A.cpu(), x.cpu()))

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    def test_cholesky_solve_batched(self, device, dtype):
        def cholesky_solve_batch_helper(A_dims, b_dims, upper):
            b, A, L = self.cholesky_solve_test_helper(A_dims, b_dims, upper, device, dtype)
            x_exp_list = []
            for i in range(b_dims[0]):
                x_exp_list.append(torch.cholesky_solve(b[i], L[i], upper=upper))
            x_exp = torch.stack(x_exp_list)  # Stacked output
            x_act = torch.cholesky_solve(b, L, upper=upper)  # Actual output
            self.assertEqual(x_act, x_exp)  # Equality check
            Ax = np.matmul(A.cpu(), x_act.cpu())
            self.assertEqual(b, Ax)  # Correctness check

        for upper, batchsize in itertools.product([True, False], [1, 3, 4]):
            cholesky_solve_batch_helper((5, batchsize), (batchsize, 5, 10), upper)

    @slowTest
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    def test_cholesky_solve_batched_many_batches(self, device, dtype):
        for A_dims, b_dims in zip([(5, 256, 256), (5,)], [(5, 10), (512, 512, 5, 10)]):
            for upper in [True, False]:
                b, A, L = self.cholesky_solve_test_helper(A_dims, b_dims, upper, device, dtype)
                x = torch.cholesky_solve(b, L, upper)
                Ax = torch.matmul(A, x)
                self.assertEqual(Ax, b.expand_as(Ax))

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    def test_cholesky_solve_batched_broadcasting(self, device, dtype):
        from numpy.linalg import solve
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        def run_test(A_dims, b_dims, upper):
            A_matrix_size = A_dims[-1]
            A_batch_dims = A_dims[:-2]
            A = random_hermitian_pd_matrix(A_matrix_size, *A_batch_dims,
                                           dtype=dtype, device='cpu')
            b = torch.randn(*b_dims, dtype=dtype, device='cpu')
            x_exp = torch.tensor(solve(A.numpy(), b.numpy()), dtype=dtype, device=device)
            A, b = A.to(dtype=dtype, device=device), b.to(dtype=dtype, device=device)
            L = torch.linalg.cholesky(A, upper=upper)
            x = torch.cholesky_solve(b, L, upper=upper)
            self.assertEqual(x, x_exp)
            # https://github.com/pytorch/pytorch/issues/42695
            x = torch.cholesky_solve(b, L, upper=upper, out=x)
            self.assertEqual(x, x_exp)

        # test against numpy.linalg.solve
        for upper in [True, False]:
            run_test((2, 1, 3, 4, 4), (2, 1, 3, 4, 6), upper)  # no broadcasting
            run_test((2, 1, 3, 4, 4), (4, 6), upper)  # broadcasting b
            run_test((4, 4), (2, 1, 3, 4, 2), upper)  # broadcasting A
            run_test((1, 3, 1, 4, 4), (2, 1, 3, 4, 5), upper)  # broadcasting A & b

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_cholesky_solve_out_errors_and_warnings(self, device, dtype):
        # dtypes should be safely castable
        a = torch.eye(2, dtype=dtype, device=device)
        b = torch.randn(2, 1, dtype=dtype, device=device)
        out = torch.empty(0, dtype=torch.int, device=device)
        with self.assertRaisesRegex(RuntimeError, "but got result with dtype Int"):
            torch.cholesky_solve(b, a, out=out)

        # device should match
        if torch.cuda.is_available():
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            out = torch.empty(0, dtype=dtype, device=wrong_device)
            with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                torch.cholesky_solve(b, a, out=out)

        # if out tensor with wrong shape is passed a warning is given
        with warnings.catch_warnings(record=True) as w:
            out = torch.empty(1, dtype=dtype, device=device)
            # Trigger warning
            torch.cholesky_solve(b, a, out=out)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

    

instantiate_device_type_tests(TestLinalg, globals())

if __name__ == '__main__':
    run_tests()
