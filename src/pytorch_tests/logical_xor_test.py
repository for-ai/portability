# Owner(s): ["module: tests"]

import torch
import numpy as np

import itertools
from itertools import product
import math
import random
from numbers import Number
import unittest
import warnings
import operator
from functools import partial

import torch.autograd.forward_ad as fwAD
from torch._six import inf, nan
from torch.testing._internal.common_utils import (
    TestCase,
    slowTest,
    iter_indices,
    TEST_WITH_ASAN,
    run_tests,
    gradcheck,
    torch_to_numpy_dtype_dict,
    numpy_to_torch_dtype_dict,
    TEST_SCIPY,
    set_default_dtype,
)
from torch.testing._internal.common_device_type import (
    expectedFailureMeta,
    instantiate_device_type_tests,
    onlyCUDA,
    onlyCPU,
    dtypes,
    dtypesIfCUDA,
    dtypesIfCPU,
    deviceCountAtLeast,
    precisionOverride,
    onlyNativeDeviceTypes,
    skipIf,
    ops,
    OpDTypes,
    skipMeta,
)
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import (
    all_types_and_complex_and,
    all_types_and,
    integral_types,
    complex_types,
    integral_types_and,
    floating_types_and,
    floating_and_complex_types,
    get_all_math_dtypes,
)
from torch.testing._internal.common_methods_invocations import (
    binary_ufuncs,
    binary_ufuncs_and_refs,
    generate_elementwise_binary_tensors,
    generate_elementwise_binary_small_value_tensors,
    generate_elementwise_binary_large_value_tensors,
    generate_elementwise_binary_extremal_value_tensors,
    generate_elementwise_binary_broadcasting_tensors,
    generate_elementwise_binary_with_scalar_samples,
)

if TEST_SCIPY:
    import scipy.special
    import scipy.integrate

# TODO: update to use opinfos consistently
class TestBinaryUfuncs(TestCase):


    def test_logical_xor_with_nontrivial_alignment(self, device):
        # test tensor that is not aligned to multiple of 16 bytes
        size = 128
        a = torch.randn(size, device=device) > 0
        b = torch.randn(size, device=device) > 0
        c = torch.randn(size, device=device) > 0
        non_trivial_alignment = [1, 2, 4, 8, 15]
        for i in non_trivial_alignment:
            for j in non_trivial_alignment:
                for k in non_trivial_alignment:
                    a_ = a[i : 100 + i]
                    b_ = b[j : 100 + j]
                    c_ = c[k : 100 + k]
                    torch.logical_xor(a_, b_, out=c_)
                    for x, y, z in zip(a_.tolist(), b_.tolist(), c_.tolist()):
                        self.assertEqual(x ^ y, z)

    def _generate_input(self, shape, dtype, device, with_extremal):
        if shape == ():
            x = torch.tensor((), dtype=dtype, device=device)
        else:
            if dtype.is_floating_point or dtype.is_complex:
                # work around torch.randn not being implemented for bfloat16
                if dtype == torch.bfloat16:
                    x = torch.randn(*shape, device=device) * random.randint(30, 100)
                    x = x.to(torch.bfloat16)
                else:
                    x = torch.randn(
                        *shape, dtype=dtype, device=device
                    ) * random.randint(30, 100)
                x[torch.randn(*shape) > 0.5] = 0
                if with_extremal and dtype.is_floating_point:
                    # Use extremal values
                    x[torch.randn(*shape) > 0.5] = float("nan")
                    x[torch.randn(*shape) > 0.5] = float("inf")
                    x[torch.randn(*shape) > 0.5] = float("-inf")
                elif with_extremal and dtype.is_complex:
                    x[torch.randn(*shape) > 0.5] = complex("nan")
                    x[torch.randn(*shape) > 0.5] = complex("inf")
                    x[torch.randn(*shape) > 0.5] = complex("-inf")
            elif dtype == torch.bool:
                x = torch.zeros(shape, dtype=dtype, device=device)
                x[torch.randn(*shape) > 0.5] = True
            else:
                x = torch.randint(15, 100, shape, dtype=dtype, device=device)

        return x


    @dtypes(
        *tuple(
            itertools.combinations_with_replacement(
                all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool), 2
            )
        )
    )
    def test_comparison_ops_type_promotion_and_broadcasting(self, device, dtypes):
        # issue #42660
        # testing all combinations of broadcasting and type promotion
        # with a range of dtypes and input shapes, and with extremal values
        def compare_with_numpy_bin_op(torch_fn, np_fn, x, y, out=None):
            # working around the fact that numpy doesn't support bfloat16
            # by letting numpy treat them as float32's
            x_np = x if x.dtype != torch.bfloat16 else x.to(torch.float32)
            y_np = (
                y.cpu().numpy()
                if y.dtype != torch.bfloat16
                else y.to(torch.float32).cpu().numpy()
            )
            self.compare_with_numpy(
                lambda inp: torch_fn(inp, y, out=out) if out else torch_fn(inp, y),
                lambda inp: np_fn(inp, y_np, out=out) if out else np_fn(inp, y_np),
                x_np,
            )

        complex_op_denylist = [
            torch.lt,
            torch.le,
            torch.gt,
            torch.ge,
        ]  # complex not supported
        input_sizes = [(1,), (10,), (10, 1), (1, 10), (4, 10), (64, 10), (12, 3)]
        op_pairs = [
            (torch.lt, np.less),
            (torch.le, np.less_equal),
            (torch.gt, np.greater),
            (torch.ge, np.greater_equal),
            (torch.eq, np.equal),
            (torch.ne, np.not_equal),
            (torch.logical_and, np.logical_and),
            (torch.logical_or, np.logical_or),
            (torch.logical_xor, np.logical_xor),
        ]

        for size1 in input_sizes:
            size2 = (2,) + size1  # perform broadcasting
            for with_extremal in [False, True]:
                a = self._generate_input(size1, dtypes[0], device, with_extremal)
                b = self._generate_input(size2, dtypes[1], device, with_extremal)
                for torch_op, numpy_op in op_pairs:
                    if (
                        dtypes[0].is_complex or dtypes[1].is_complex
                    ) and torch_op in complex_op_denylist:
                        continue
                    # functional version of op
                    compare_with_numpy_bin_op(torch_op, numpy_op, a, b)

                    # functional comparison ops always return bool tensors
                    self.assertEqual(torch_op(a, b).dtype, torch.bool)

                    # out version of op
                    out = torch.zeros(
                        1, dtype=torch.complex128
                    )  # all casts to complex128 are safe
                    compare_with_numpy_bin_op(torch_op, numpy_op, a, b, out=out)

def generate_not_implemented_tests(cls):
    class UnknownType:
        pass

    # TODO: refactor to inline these
    _types = [
        torch.half,
        torch.float,
        torch.double,
        torch.int8,
        torch.short,
        torch.int,
        torch.long,
        torch.uint8,
    ]

    def create_test_func(op):
        @dtypes(*_types)
        def test(self, device, dtype):
            # Generate the inputs
            tensor = torch.empty((), device=device, dtype=dtype)

            # Runs the tensor op on the device
            result = getattr(tensor, op)(UnknownType())
            self.assertEqual(result, NotImplemented)

        return test

    for op in tensor_binary_ops:
        test_name = "test_{}_not_implemented".format(op)
        assert not hasattr(cls, test_name), "{0} already in {1}".format(
            test_name, cls.__name__
        )

        setattr(cls, test_name, create_test_func(op))
tensor_binary_ops = [
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
    "__eq__",
    "__ne__",
    "__add__",
    "__radd__",
    "__iadd__",
    "__sub__",
    "__rsub__",
    "__isub__",
    "__mul__",
    "__rmul__",
    "__imul__",
    "__matmul__",
    "__rmatmul__",
    "__truediv__",
    "__rtruediv__",
    "__itruediv__",
    "__floordiv__",
    "__rfloordiv__",
    "__ifloordiv__",
    "__mod__",
    "__rmod__",
    "__imod__",
    "__pow__",
    "__rpow__",
    "__ipow__",
    "__lshift__",
    "__rlshift__",
    "__ilshift__",
    "__rshift__",
    "__rrshift__",
    "__irshift__",
    "__and__",
    "__rand__",
    "__iand__",
    "__xor__",
    "__rxor__",
    "__ixor__",
    "__or__",
    "__ror__",
    "__ior__",
    # Unsupported operators
    # '__imatmul__',
    # '__divmod__', '__rdivmod__', '__idivmod__',
]

generate_not_implemented_tests(TestBinaryUfuncs)
instantiate_device_type_tests(TestBinaryUfuncs, globals())

if __name__ == "__main__":
    run_tests()
