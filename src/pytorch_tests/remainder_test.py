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
    def _test_cop(self, torchfn, mathfn, dtype, device):
        def reference_implementation(res2):
            for i, j in iter_indices(sm1):
                idx1d = i * sm1.size(0) + j
                res2[i, j] = mathfn(sm1[i, j], sm2[idx1d])
            return res2

        # contiguous
        m1 = torch.randn(10, 10, 10, dtype=dtype, device=device)
        m2 = torch.randn(10, 10 * 10, dtype=dtype, device=device)
        sm1 = m1[4]
        sm2 = m2[4]

        res1 = torchfn(sm1, sm2.view(10, 10))
        res2 = reference_implementation(res1.clone())
        self.assertEqual(res1, res2)

        # non-contiguous
        m1 = torch.randn(10, 10, 10, dtype=dtype, device=device)
        m2 = torch.randn(10 * 10, 10 * 10, dtype=dtype, device=device)
        sm1 = m1[:, 4]
        sm2 = m2[:, 4]
        # view as sm1.size()
        sm2.set_(
            sm2.storage(),
            sm2.storage_offset(),
            sm1.size(),
            (sm2.stride()[0] * 10, sm2.stride()[0]),
        )
        res1 = torchfn(sm1, sm2)
        # reference_implementation assumes 1-d sm2
        sm2.set_(
            sm2.storage(), sm2.storage_offset(), m2[:, 4].size(), m2[:, 4].stride()
        )
        res2 = reference_implementation(res1.clone())
        self.assertEqual(res1, res2)


    @dtypes(*floating_types_and(torch.half))
    def test_fmod_remainder_by_zero_float(self, device, dtype):
        fn_list = (torch.fmod, torch.remainder)
        for fn in fn_list:
            # check floating-point tensor fmod/remainder to zero is nan on both CPU and GPU
            x = make_tensor((10, 10), device=device, dtype=dtype, low=-9, high=9)
            zero = torch.zeros_like(x)
            self.assertTrue(torch.all(fn(x, 0.0).isnan()))
            self.assertTrue(torch.all(fn(x, zero).isnan()))

    @onlyNativeDeviceTypes  # Check Issue https://github.com/pytorch/pytorch/issues/48130
    @dtypes(*integral_types())
    def test_fmod_remainder_by_zero_integral(self, device, dtype):
        fn_list = (torch.fmod, torch.remainder)
        for fn in fn_list:
            # check integral tensor fmod/remainder to zero
            x = make_tensor((10, 10), device=device, dtype=dtype, low=-9, high=9)
            zero = torch.zeros_like(x)
            # RuntimeError on CPU
            if self.device_type == "cpu":
                with self.assertRaisesRegex(RuntimeError, "ZeroDivisionError"):
                    fn(x, zero)
            elif torch.version.hip is not None:
                # ROCm behavior: x % 0 is a no-op; x is returned
                self.assertEqual(fn(x, zero), x)
            else:
                # CUDA behavior: Different value for different dtype
                # Due to it's an undefined behavior, CUDA returns a pattern of all 1s
                # for integral dividend (other than int64) divided by zero. For int64,
                # CUDA returns all 1s for negative dividend, half 1s for positive dividend.
                # uint8: 0xff -> 255
                # int32: 0xffffffff -> -1
                if dtype == torch.int64:
                    self.assertEqual(fn(x, zero) == 4294967295, x >= 0)
                    self.assertEqual(fn(x, zero) == -1, x < 0)
                else:
                    value = 255 if dtype == torch.uint8 else -1
                    self.assertTrue(torch.all(fn(x, zero) == value))

    @dtypes(*all_types_and(torch.half))
    def test_fmod_remainder(self, device, dtype):
        # Use numpy as reference
        def _helper(x, mod, fns_list):
            for fn, inplace_fn, ref_fn in fns_list:
                np_x = x.cpu().numpy() if torch.is_tensor(x) else x
                np_mod = mod.cpu().numpy() if torch.is_tensor(mod) else mod
                exp = ref_fn(np_x, np_mod)
                exp = torch.from_numpy(exp)
                res = fn(x, mod)

                self.assertEqual(res, exp, exact_dtype=False)

                if torch.is_tensor(x):
                    # out
                    out = torch.empty(0, device=device, dtype=res.dtype)
                    fn(x, mod, out=out)
                    self.assertEqual(out, exp, exact_dtype=False)
                    self.assertEqual(out.size(), torch.Size([10, 10]))
                    # in-place (Type cast runtime error)
                    try:
                        inplace_fn(x, mod)
                        self.assertEqual(x, exp, exact_dtype=False)
                    except RuntimeError as e:
                        self.assertRegex(
                            str(e),
                            "result type (Half|Float|Double) "
                            "can't be cast to the desired output "
                            "type (Byte|Char|Short|Int|Long)",
                        )

        x = make_tensor((10, 10), device=device, dtype=dtype, low=-9, high=9)
        # mod with same dtype as x
        mod = make_tensor((10, 10), device=device, dtype=dtype, low=-9, high=9)
        # Exclude 0
        mod[mod == 0] = 1

        # Mods: Integer, Float, Tensor, Non-contiguous Tensor
        mods = [3, 2.3, mod, mod.t()]
        # mod with floating-point dtype
        if dtype in integral_types():
            mod_float = make_tensor(
                (10, 10), device=device, dtype=torch.float, low=-9, high=9
            )
            mod[mod == 0] = 1
            mods.append(mod_float)

        for dividend, mod in product([x, x.t()], mods):
            _helper(
                dividend,
                mod,
                (
                    (torch.fmod, torch.Tensor.fmod_, np.fmod),
                    (torch.remainder, torch.Tensor.remainder_, np.remainder),
                ),
            )

        # Tests for torch.remainder(scalar, tensor)
        for dividend, mod in product([5, 3.14], mods):
            if torch.is_tensor(mod):
                _helper(
                    dividend,
                    mod,
                    ((torch.remainder, torch.Tensor.remainder_, np.remainder),),
                )

    @dtypes(torch.float, torch.double)
    def test_remainder_fmod_large_dividend(self, device, dtype):
        alarge = 1e9
        pi = 3.14159265358979
        for avalue in [alarge, -alarge]:
            for bvalue in [pi, -pi]:
                a = torch.tensor([avalue], dtype=dtype, device=device)
                b = torch.tensor([bvalue], dtype=dtype, device=device)
                c = torch.remainder(a, b)
                d = torch.fmod(a, b)
                self.assertTrue(
                    (b[0] > 0) == (c[0] > 0)
                )  # remainder has same sign as divisor
                self.assertTrue(
                    (a[0] > 0) == (d[0] > 0)
                )  # fmod has same sign as dividend
                self.assertTrue(
                    abs(c[0]) < abs(b[0])
                )  # remainder is within range of divisor
                self.assertTrue(
                    abs(d[0]) < abs(b[0])
                )  # fmod is within range of divisor
                if (a[0] > 0) == (b[0] > 0):
                    self.assertTrue(c[0] == d[0])  # remainder is same as fmod
                else:
                    self.assertTrue(
                        abs(c[0] - d[0]) == abs(b[0])
                    )  # differ by one divisor

    
    @onlyCPU
    @dtypes(torch.float)
    def test_cremainder(self, device, dtype):
        self._test_cop(torch.remainder, lambda x, y: x % y, dtype, device)

instantiate_device_type_tests(TestBinaryUfuncs, globals())

if __name__ == "__main__":
    run_tests()
