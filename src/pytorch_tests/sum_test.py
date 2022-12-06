# Owner(s): ["module: tests"]

import torch
import numpy as np

import math
from typing import Dict, List, Sequence
import random
from functools import partial
from itertools import product, combinations, permutations
import warnings

from torch._six import inf, nan
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import (
    all_types_and_complex_and, get_all_math_dtypes, integral_types, complex_types, floating_types_and,
    integral_types_and, floating_and_complex_types_and, all_types_and,
)
from torch.testing._internal.common_utils import (
    TestCase, run_tests, skipIfNoSciPy, slowTest, torch_to_numpy_dtype_dict,
    IS_WINDOWS)
from torch.testing._internal.common_device_type import (
    OpDTypes, expectedFailureMeta, instantiate_device_type_tests, onlyCPU, dtypes, dtypesIfCUDA, dtypesIfCPU,
    onlyNativeDeviceTypes, onlyCUDA, largeTensorTest, ops, precisionOverride)
from torch.testing._internal.common_methods_invocations import (
    ReductionOpInfo, reduction_ops, reference_masked_ops)


class TestReductions(TestCase):
    # TODO: update this and tests that use it to use the device argument properly
    def _assert_matches_numpy(self, t, n):
        self.assertEqual(n.shape, t.shape)
        if t.dtype == torch.float:
            self.assertEqual(n, t, rtol=1e-03, atol=1e-05, equal_nan=True)
        else:
            self.assertEqual(n, t, equal_nan=True)

    def _make_tensors(self, shape, val_range=(-100, 100), use_floating=True, use_integral=True,
                      use_complex=False) -> Dict[str, List[torch.Tensor]]:
        float_types = [torch.double,
                       torch.float]
        int_types = [torch.int64,
                     torch.int32,
                     torch.int16]

        complex_types = [torch.complex64,
                         torch.complex128]

        def make_contiguous(shape, dtype) -> torch.Tensor:
            if dtype in float_types:
                val = torch.randn(shape, dtype=dtype)
                val = val * ((val_range[1] - val_range[0]) / (math.pi * 2.0))
                val = val + ((val_range[1] - val_range[0]) / 2.0)
                val = torch.clamp(val, min=val_range[0], max=val_range[1])
                return val
            result = torch.zeros(shape, dtype=dtype)
            result.apply_(lambda x: random.randint(val_range[0], val_range[1]))
            return result

        def make_non_contiguous(shape, dtype) -> torch.Tensor:
            contig = make_contiguous(shape, dtype)
            non_contig = torch.empty(shape + (2, 2), dtype=dtype)[..., 0]
            non_contig = non_contig.select(-1, -1)
            non_contig.copy_(contig)
            self.assertFalse(non_contig.is_contiguous())
            return non_contig

        def make_contiguous_slice(size, dtype) -> torch.Tensor:
            contig = make_contiguous((1, size), dtype)
            non_contig = contig[:1, 1:size - 1]
            self.assertTrue(non_contig.is_contiguous())
            return contig

        types = []
        if use_floating:
            types += float_types
        if use_integral:
            types += int_types
        if use_complex:
            types += complex_types
        tensors: Dict[str, List[torch.Tensor]] = {
            "cont": [], "noncont": [], "slice": []}
        for dtype in types:
            tensors["cont"].append(make_contiguous(shape, dtype))
            tensors["noncont"].append(make_non_contiguous(shape, dtype))
            tensors["slice"].append(
                make_contiguous_slice(sum(list(shape)), dtype))

        return tensors

    def _test_dim_ops(self, pytorch_op, numpy_op,
                      use_floating=True, use_integral=True, use_complex=False):
        def do_one(tensors_dict, dim):
            for category, tensors in tensors_dict.items():
                if category == "slice":
                    dim = 0
                for tensor in tensors:
                    # we have no control over NumPy warnings...
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        expected = numpy_op(tensor.cpu().numpy(), dim)
                    actual = pytorch_op(tensor, dim)
                    self._assert_matches_numpy(actual, expected)
                    if torch.cuda.is_available():
                        self._assert_matches_numpy(pytorch_op(
                            tensor.cuda(), dim).cpu(), expected)
        do_one(self._make_tensors((5, 400000), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 1)
        do_one(self._make_tensors((3, 5, 7), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 0)
        do_one(self._make_tensors((3, 5, 7), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 1)
        do_one(self._make_tensors((3, 5, 7), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 2)
        do_one(self._make_tensors((100000, ), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), -1)
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 0)
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 1)
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 2)
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), (1, 2))
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), (1, -1))
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), (0, 2))
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), (0, 2, 1))

    @onlyCPU
    def test_sum_dim(self, device):
        self._test_dim_ops(
            lambda t, d: t.sum(d),
            lambda n, d: n.sum(d),
            use_floating=True, use_integral=True, use_complex=True)


instantiate_device_type_tests(TestReductions, globals())

if __name__ == '__main__':
    run_tests()
