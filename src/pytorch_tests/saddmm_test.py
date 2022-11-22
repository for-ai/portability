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

def _sparse_to_dense(tensor):
    if tensor.dtype != torch.bool:
        return tensor.to_dense()

    # to_dense uses coalesce which isn't implemented for bool
    return tensor.to(torch.int8).to_dense().to(torch.bool)


class TestSparseBase(TestCase):
    def run(self, result=None):
        if TEST_WITH_CROSSREF:
            with CrossRefSparseFakeMode():
                return super().run(result)
        else:
            return super().run(result)


class TestSparse(TestSparseBase):

    def setUp(self):
        TestCase.setUp(self)

        self.index_tensor = lambda *args, **kwargs: torch.tensor(*args, **kwargs, dtype=torch.int64)

        def sparse_empty_factory(*args, **kwargs):
            kwargs['layout'] = kwargs.get('layout', torch.sparse_coo)
            return torch.empty(*args, **kwargs)
        self.sparse_empty = sparse_empty_factory

        def sparse_tensor_factory(*args, **kwargs):
            return torch.sparse_coo_tensor(*args, **kwargs)
        self.sparse_tensor = sparse_tensor_factory
        self.legacy_sparse_tensor = torch.sparse.DoubleTensor

    def _gen_sparse(self, sparse_dim, nnz, with_size, dtype, device, coalesced):
        if isinstance(with_size, Number):
            with_size = [with_size] * sparse_dim

        x, i, v = self.genSparseTensor(with_size, sparse_dim, nnz, not coalesced, dtype=dtype, device=device)

        if not coalesced:
            self.assert_uncoalesced(x)

        return x, i, v


    def assert_uncoalesced(self, x):
        """
        Test if a CPU tensor is uncoalesced.  This is used to ensure
        correctness of the uncoalesced tensor generation algorithm.
        """
        assert not x.is_coalesced()
        existing_indices = set()
        for i in range(x._nnz()):
            index = str(x._indices()[:, i])
            if index in existing_indices:
                return True
            else:
                existing_indices.add(index)

    @onlyCPU
    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_saddmm(self, device, dtype, coalesced):
        def test_shape(di, dj, dk, nnz):
            x = self._gen_sparse(2, nnz, [di, dj], dtype, device, coalesced)[0]
            t = self._gen_sparse(2, nnz, [di, dk], dtype, device, coalesced)[0]
            y = torch.randn(dj, dk, dtype=dtype, device=device)
            alpha = random.random()
            beta = random.random()

            res = torch.saddmm(t, x, y, beta=beta, alpha=alpha)
            expected = torch.addmm(self.safeToDense(t), self.safeToDense(x), y, beta=beta, alpha=alpha)
            self.assertEqual(self.safeToDense(res), expected)

            res = torch.saddmm(t, x, y)
            expected = torch.addmm(self.safeToDense(t), self.safeToDense(x), y)
            self.assertEqual(self.safeToDense(res), expected)

            res = torch.smm(x, y)
            expected = torch.mm(self.safeToDense(x), y)
            self.assertEqual(self.safeToDense(res), expected)

        test_shape(7, 5, 3, 20)
        test_shape(1000, 100, 100, 20)
        test_shape(3000, 64, 300, 20)
        test_shape(0, 100, 100, 0)
        test_shape(1000, 0, 100, 0)
        test_shape(1000, 100, 0, 0)


instantiate_device_type_tests(TestSparse, globals(), except_for='meta')
if __name__ == '__main__':
    run_tests()
