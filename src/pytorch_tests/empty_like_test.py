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

    def _assert_sparse_invars(self, t):
        # SparseTensor has the following invariants:
        # - sparse_dim + dense_dim = len(SparseTensor.shape)
        # - SparseTensor._indices().shape = (sparse_dim, nnz)
        # - SparseTensor._values().shape = (nnz, SparseTensor.shape[sparse_dim:])
        self.assertEqual(t.sparse_dim() + t.dense_dim(), len(t.shape))
        self.assertEqual(tuple(t._indices().shape), (t.sparse_dim(), t._nnz()))
        self.assertEqual(tuple(t._values().shape), (t._nnz(), ) + t.shape[t.sparse_dim():])


    def _test_empty_like(self, sparse_tensor, dtype, device, coalesced):

        result = torch.empty_like(sparse_tensor)
        self.assertTrue(result.is_sparse)
        self._assert_sparse_invars(result)
        self.assertEqual(result.shape, sparse_tensor.shape)
        self.assertEqual(result.dtype, sparse_tensor.dtype)
        self.assertEqual(result.device, sparse_tensor.device)
        self.assertEqual(result.sparse_dim(), sparse_tensor.sparse_dim())
        self.assertEqual(result.dense_dim(), sparse_tensor.dense_dim())

        sparse_tensor, _, _ = self._gen_sparse(len([2, 3]), 9, [2, 3] + [5, 6], dtype, device, coalesced)
        data = (sparse_tensor, sparse_tensor, sparse_tensor, sparse_tensor.unsqueeze(0))
        mem_formats = [torch.channels_last, torch.contiguous_format, torch.preserve_format, torch.channels_last_3d]
        for x, mem_format in zip(data, mem_formats):

            with self.assertRaisesRegex(RuntimeError, "memory format option is only supported by strided tensors"):
                result = torch.empty_like(x, memory_format=mem_format)

            result = torch.empty_like(x, layout=torch.strided, memory_format=mem_format)
            self.assertTrue(result.layout == torch.strided)

        with self.assertRaisesRegex(
            RuntimeError, r"Could not run 'aten::empty_strided' with arguments from the 'Sparse(CPU|CUDA)' backend"
        ):
            dense_tensor = sparse_tensor.to_dense()
            result = torch.empty_like(dense_tensor, layout=torch.sparse_coo)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_empty_like(self, device, dtype, coalesced):
        # tests https://github.com/pytorch/pytorch/issues/43699

        if coalesced:
            input_coalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[0, 1, 2]]),
                values=torch.tensor([3.0, -4.0, 5.0]),
                size=[3, ],
                dtype=dtype,
                device=device
            ).coalesce()
            self._test_empty_like(input_coalesced, dtype, device, coalesced)

            # hybrid sparse input
            input_coalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[1, 3], [2, 4]]),
                values=torch.tensor([[-1.0, 3.0], [-5.0, 7.0]]),
                size=[4, 5, 2],
                dtype=dtype,
                device=device
            ).coalesce()
            self._test_empty_like(input_coalesced, dtype, device, coalesced)

        if not coalesced:
            # test uncoalesced input
            input_uncoalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[0], [1], [2], [0], [1], [2]]).transpose(1, 0),
                values=torch.tensor([2.0, -3.0, -4.0, 1.0, -1.0, 1.5]),
                size=[3, ],
                dtype=dtype,
                device=device
            )
            self._test_empty_like(input_uncoalesced, dtype, device, coalesced)

            # test on empty sparse tensor
            input_uncoalesced = torch.sparse_coo_tensor(
                indices=torch.zeros([2, 0]),
                values=torch.zeros([0, 5, 5, 5, 5, 5, 5, 0]),
                size=[0, 0, 5, 5, 5, 5, 5, 5, 0],
                dtype=dtype,
                device=device
            )
            self._test_empty_like(input_uncoalesced, dtype, device, coalesced)

    

# e.g., TestSparseCPU and TestSparseCUDA
instantiate_device_type_tests(TestSparse, globals(), except_for='meta')

if __name__ == '__main__':
    run_tests()
