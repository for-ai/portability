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
from ..utils.pytorch_device_decorators import onlyNativeDeviceTypes, onlyAcceleratedDeviceTypes, instantiate_device_type_tests
from ..utils.timer_wrapper import pytorch_op_timer

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

    @skipCPUIfNoLapack
    @skipCUDAIfNoCusolver
    @dtypes(*floating_and_complex_types())
    def test_ormqr(self, device, dtype):

        def run_test(batch, m, n, fortran_contiguous):
            A = make_tensor((*batch, m, n), dtype=dtype, device=device)
            reflectors, tau = torch.geqrf(A)
            if not fortran_contiguous:
                self.assertTrue(reflectors.mT.is_contiguous())
                reflectors = reflectors.contiguous()

            # Q is of size m x m
            Q, _ = torch.linalg.qr(A, mode='complete')
            C_right = make_tensor((*batch, m, n), dtype=dtype, device=device)
            C_left = make_tensor((*batch, n, m), dtype=dtype, device=device)

            expected = Q @ C_right
            actual = torch.ormqr(reflectors, tau, C_right, left=True, transpose=False)
            self.assertEqual(expected, actual)

            expected = C_left @ Q
            actual = torch.ormqr(reflectors, tau, C_left, left=False, transpose=False)
            self.assertEqual(expected, actual)

            expected = Q.mH @ C_right
            actual = torch.ormqr(reflectors, tau, C_right, left=True, transpose=True)
            self.assertEqual(expected, actual)

            expected = C_left @ Q.mH
            actual = torch.ormqr(reflectors, tau, C_left, left=False, transpose=True)
            self.assertEqual(expected, actual)

            # if tau is all zeros then the implicit matrix Q is the identity matrix
            # so the actual result should be C_right in this case
            zero_tau = torch.zeros_like(tau)
            actual = torch.ormqr(reflectors, zero_tau, C_right, left=True, transpose=False)
            self.assertEqual(C_right, actual)

        batches = [(), (0, ), (2, ), (2, 1)]
        ns = [5, 2, 0]
        for batch, (m, n), fortran_contiguous in product(batches, product(ns, ns), [True, False]):
            run_test(batch, m, n, fortran_contiguous)

    @skipCPUIfNoLapack
    @skipCUDAIfNoCusolver
    @dtypes(*floating_and_complex_types())
    def test_ormqr_errors_and_warnings(self, device, dtype):
        test_cases = [
            # input1 size, input2 size, input3 size, error regex
            ((10,), (2,), (2,), r"input must have at least 2 dimensions"),
            ((2, 2), (2,), (2,), r"other must have at least 2 dimensions"),
            ((10, 6), (20,), (10, 6), r"other.shape\[-2\] must be greater than or equal to tau.shape\[-1\]"),
            ((6, 6), (5,), (5, 5), r"other.shape\[-2\] must be equal to input.shape\[-2\]"),
            ((1, 2, 2), (2, 2), (1, 2, 2), r"batch dimensions of tau to be equal to input.shape\[:-2\]"),
            ((1, 2, 2), (1, 2), (2, 2, 2), r"batch dimensions of other to be equal to input.shape\[:-2\]"),
        ]
        for a_size, tau_size, c_size, error_regex in test_cases:
            a = make_tensor(a_size, dtype=dtype, device=device)
            tau = make_tensor(tau_size, dtype=dtype, device=device)
            c = make_tensor(c_size, dtype=dtype, device=device)
            with self.assertRaisesRegex(RuntimeError, error_regex):
                torch.ormqr(a, tau, c)

    
instantiate_device_type_tests(TestLinalg, globals())

if __name__ == '__main__':
    run_tests()
