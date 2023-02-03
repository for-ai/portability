# Owner(s): ["module: sparse"]

import torch
import itertools
import functools
import operator
import random
import unittest
from torch.testing import make_tensor
from torch.testing._internal.common_utils import TestCase, run_tests, skipIfRocm, do_test_dtypes, \
    do_test_empty_full, load_tests, TEST_NUMPY, TEST_SCIPY, IS_WINDOWS, gradcheck, coalescedonoff, \
    DeterministicGuard, first_sample, TEST_WITH_CROSSREF, TEST_WITH_ROCM
from torch.testing._internal.common_cuda import TEST_CUDA, _get_torch_cuda_version
from numbers import Number
from typing import Dict, Any
from distutils.version import LooseVersion
from torch.testing._internal.common_cuda import \
    (SM53OrLater, SM80OrLater, CUDA11OrLater)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, ops, dtypes, dtypesIfCUDA, onlyCPU, onlyCUDA, precisionOverride,
     deviceCountAtLeast, OpDTypes)
from torch.testing._internal.common_methods_invocations import \
    (sparse_unary_ufuncs, sparse_masked_reduction_ops)
from torch.testing._internal.common_dtype import (
    all_types, all_types_and_complex, all_types_and_complex_and, floating_and_complex_types,
    floating_and_complex_types_and, integral_types, floating_types_and,
)

if TEST_SCIPY:
    import scipy.sparse

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

# batched grad doesn't support sparse
gradcheck = functools.partial(gradcheck, check_batched_grad=False)

CUSPARSE_SPMM_COMPLEX128_SUPPORTED = (
    IS_WINDOWS and torch.version.cuda and LooseVersion(torch.version.cuda) > "11.2"
) or (not IS_WINDOWS and CUDA11OrLater)

class TestSparseBase(TestCase):
    def run(self, result=None):
        if TEST_WITH_CROSSREF:
            with CrossRefSparseFakeMode():
                return super().run(result)
        else:
            return super().run(result)

class TestSparse(TestSparseBase):

    def _test_log1p_tensor(self, sparse_tensor, coalesced):
        def is_integral(dtype):
            return dtype in integral_types()

        dense_tensor = sparse_tensor.to_dense()
        expected_output = dense_tensor.log1p()
        is_integral_dtype = is_integral(sparse_tensor.dtype)
        self.assertEqual(expected_output, sparse_tensor.log1p().to_dense())
        if is_integral_dtype:
            with self.assertRaisesRegex(RuntimeError, "result type .* can't be cast to"):
                sparse_tensor.coalesce().log1p_()
        else:
            self.assertEqual(expected_output, sparse_tensor.coalesce().log1p_().to_dense())

        if not coalesced:
            # test in-place op on uncoalesced input
            with self.assertRaisesRegex(RuntimeError, "log1p_ requires coalesced input"):
                sparse_tensor.log1p_()

        if is_integral_dtype:
            with self.assertRaisesRegex(RuntimeError, "only Tensors of floating point dtype can require gradients"):
                sparse_tensor.requires_grad_()

    @coalescedonoff
    @dtypes(*all_types())
    def test_log1p(self, device, dtype, coalesced):
        if coalesced:
            input_coalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[0], [1], [2]]).transpose(1, 0),
                values=torch.tensor([3.0, 4.0, 5.0]),
                size=[3, ],
                device=device,
                dtype=dtype
            ).coalesce()
            self._test_log1p_tensor(input_coalesced, coalesced)

            # hybrid sparse input
            input_coalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[1, 3], [2, 4]]),
                values=torch.tensor([[1.0, 3.0], [5.0, 7.0]]),
                size=[4, 5, 2],
                device=device,
                dtype=dtype
            ).coalesce()
            self._test_log1p_tensor(input_coalesced, coalesced)

        if not coalesced:
            # test uncoalesced input
            input_uncoalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[0], [1], [2], [0], [1], [2]]).transpose(1, 0),
                values=torch.tensor([2.0, 3.0, 4.0, 1.0, 1.0, 1.0]),
                size=[3, ],
                device=device,
                dtype=dtype
            )
            self._test_log1p_tensor(input_uncoalesced, coalesced)

            # test on empty sparse tensor
            input_uncoalesced = torch.sparse_coo_tensor(
                indices=torch.zeros([2, 0]),
                values=torch.zeros([0, 5, 5, 5, 5, 5, 5, 0]),
                size=[0, 0, 5, 5, 5, 5, 5, 5, 0],
                device=device,
                dtype=dtype
            )
            # empty tensors are coalesced at creation (nnz < 2) we must force the uncoalesced state
            input_uncoalesced._coalesced_(False)
            self._test_log1p_tensor(input_uncoalesced, coalesced)

# e.g., TestSparseCPU and TestSparseCUDA
instantiate_device_type_tests(TestSparse, globals(), except_for='meta')

if __name__ == '__main__':
    run_tests()
