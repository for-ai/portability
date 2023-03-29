import collections
import copy
import itertools
import math
import sys
from typing import Type

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

# pylint: disable=unused-import,g-bad-import-order
from tensorflow.python.framework import dtypes
from tensorflow.python.lib.core import _pywrap_bfloat16
from tensorflow.python.platform import test

bfloat16 = _pywrap_bfloat16.TF_bfloat16_type()
float8_e4m3b11 = _pywrap_bfloat16.TF_float8_e4m3b11_type()


UNARY_UFUNCS = [
    np.rad2deg
]


def numpy_assert_allclose(a, b, float_type, **kwargs):
    a = a.astype(np.float32) if a.dtype == float_type else a
    b = b.astype(np.float32) if b.dtype == float_type else b
    return np.testing.assert_allclose(a, b, **kwargs)


def truncate(x, float_type):
    if isinstance(x, np.ndarray):
        return x.astype(float_type).astype(np.float32)
    else:
        return type(x)(float_type(x))


# pylint: disable=g-complex-comprehension
@parameterized.named_parameters(({
    "testcase_name": "_" + dtype.__name__,
    "float_type": dtype
} for dtype in [bfloat16, float8_e4m3b11]))
class CustomFloatNumPyTest(parameterized.TestCase):

    def testUnaryUfunc(self, float_type):
        for op in UNARY_UFUNCS:
            with self.subTest(op.__name__):
                rng = np.random.RandomState(seed=42)
                x = rng.randn(3, 7, 10).astype(float_type)
                numpy_assert_allclose(
                    op(x).astype(np.float32),
                    truncate(op(x.astype(np.float32)), float_type=float_type),
                    rtol=1e-4,
                    float_type=float_type)


if __name__ == "__main__":
    absltest.main()
