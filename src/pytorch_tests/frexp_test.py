# Owner(s): ["module: tests"]

import torch
import numpy as np

import math
from numbers import Number
import random
import unittest
from hypothesis import given
import torch.testing._internal.hypothesis_utils as hu

from torch._six import inf, nan
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    torch_to_numpy_dtype_dict,
    numpy_to_torch_dtype_dict,
    suppress_warnings,
    TEST_SCIPY,
    slowTest,
    skipIfNoSciPy,
    IS_WINDOWS,
    gradcheck,
    TEST_WITH_ASAN,
)
from torch.testing._internal.common_methods_invocations import (
    unary_ufuncs,
    generate_elementwise_unary_tensors,
    generate_elementwise_unary_small_value_tensors,
    generate_elementwise_unary_large_value_tensors,
    generate_elementwise_unary_extremal_value_tensors,
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
    dtypes,
    onlyCPU,
    onlyNativeDeviceTypes,
    onlyCUDA,
    dtypesIfCUDA,
    precisionOverride,
    dtypesIfCPU,
)

from torch.testing import make_tensor
from torch.testing._internal.common_dtype import (
    floating_types_and,
    all_types_and_complex_and,
    integral_types_and,
    get_all_math_dtypes,
    complex_types,
    all_types_and,
    floating_and_complex_types_and,
)

if TEST_SCIPY:
    import scipy

class TestUnaryUfuncs(TestCase):
    exact_dtype = True

    @ops(
        [_fn for _fn in unary_ufuncs if _fn.domain != (None, None)],
        allowed_dtypes=floating_types_and(torch.bfloat16, torch.half),
    )
    def test_float_domains(self, device, dtype, op):
        eps = (1e-5, 1e-3, 1e-1, 1, 2, 10, 20, 50, 100)

        low, high = op.domain
        # NOTE: the following two loops are separated for readability
        if low is not None:
            low_tensor = torch.tensor(low, device=device, dtype=dtype)
            for epsilon in eps:
                lower_tensor = low_tensor - epsilon

                # Skips the test if the difference is not representable,
                #   which can occur if, for example, the difference is small
                #   and the dtype is imprecise (like bfloat16 is)
                if lower_tensor.item() == low_tensor.item():
                    continue

                result = op(lower_tensor)
                self.assertEqual(
                    result.item(),
                    float("nan"),
                    msg=(
                        "input of {0} outside lower domain boundary"
                        " {1} produced {2}, not nan!"
                    ).format(lower_tensor.item(), low, result.item()),
                )

        if high is not None:
            high_tensor = torch.tensor(high, device=device, dtype=dtype)
            for epsilon in eps:
                higher_tensor = high_tensor + epsilon

                # See above comment
                if higher_tensor.item() == high_tensor.item():
                    continue

                result = op(higher_tensor)
                self.assertEqual(
                    result.item(),
                    float("nan"),
                    msg=(
                        "input of {0} outside upper domain boundary"
                        " {1} produced {2}, not nan!"
                    ).format(higher_tensor.item(), high, result.item()),
                )

    # Helper for comparing torch tensors and numpy arrays
    # TODO: should this or assertEqual also validate that strides are equal?
    def assertEqualHelper(
        self, actual, expected, msg, *, dtype, exact_dtype=True, **kwargs
    ):
        assert isinstance(actual, torch.Tensor)

        # Some NumPy functions return scalars, not arrays
        if isinstance(expected, Number):
            self.assertEqual(actual.item(), expected, msg, **kwargs)
        elif isinstance(expected, np.ndarray):
            # Handles exact dtype comparisons between arrays and tensors
            if exact_dtype:
                if (
                    actual.dtype is torch.bfloat16
                    or expected.dtype != torch_to_numpy_dtype_dict[actual.dtype]
                ):
                    # Allows array dtype to be float32 when comparing with bfloat16 tensors
                    #   since NumPy doesn't support the bfloat16 dtype
                    # Also ops like scipy.special.erf, scipy.special.erfc, etc, promote float16
                    # to float32
                    if expected.dtype == np.float32:
                        assert actual.dtype in (
                            torch.float16,
                            torch.bfloat16,
                            torch.float32,
                        )
                    elif expected.dtype == np.float64:
                        assert actual.dtype in (
                            torch.float16,
                            torch.bfloat16,
                            torch.float32,
                            torch.float64,
                        )
                    else:
                        self.fail(
                            "Expected dtype {0} but got {1}!".format(
                                expected.dtype, actual.dtype
                            )
                        )

            self.assertEqual(
                actual,
                torch.from_numpy(expected).to(actual.dtype),
                msg,
                exact_device=False,
                **kwargs
            )
        else:
            self.assertEqual(actual, expected, msg, exact_device=False, **kwargs)

    @dtypes(*floating_types_and(torch.half))
    def test_frexp(self, device, dtype):
        input = make_tensor((50, 50), dtype=dtype, device=device)
        mantissa, exponent = torch.frexp(input)
        np_mantissa, np_exponent = np.frexp(input.cpu().numpy())

        self.assertEqual(mantissa, np_mantissa)
        self.assertEqual(exponent, np_exponent)

        # torch.frexp returns exponent in int32 to be compatible with np.frexp
        self.assertTrue(exponent.dtype == torch.int32)
        self.assertTrue(torch_to_numpy_dtype_dict[exponent.dtype] == np_exponent.dtype)

    def test_frexp_assert_raises(self, device):
        invalid_input_dtypes = integral_types_and(torch.bool) + complex_types()
        for dtype in invalid_input_dtypes:
            input = make_tensor((50, 50), dtype=dtype, device=device)
            with self.assertRaisesRegex(
                RuntimeError, r"torch\.frexp\(\) only supports floating-point dtypes"
            ):
                torch.frexp(input)

        for dtype in floating_types_and(torch.half):
            input = make_tensor((50, 50), dtype=dtype, device=device)

            dtypes = list(
                all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16)
            )
            dtypes.remove(dtype)
            for mantissa_dtype in dtypes:
                mantissa = torch.empty_like(input, dtype=mantissa_dtype)
                exponent = torch.empty_like(input, dtype=torch.int)
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"torch\.frexp\(\) expects mantissa to have dtype .+ but got .+",
                ):
                    torch.frexp(input, out=(mantissa, exponent))

            dtypes.append(dtype)
            dtypes.remove(torch.int)
            for exponent_dtype in dtypes:
                mantissa = torch.empty_like(input)
                exponent = torch.empty_like(input, dtype=exponent_dtype)
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"torch\.frexp\(\) expects exponent to have int dtype but got .+",
                ):
                    torch.frexp(input, out=(mantissa, exponent))

    
instantiate_device_type_tests(TestUnaryUfuncs, globals())

if __name__ == "__main__":
    run_tests()
