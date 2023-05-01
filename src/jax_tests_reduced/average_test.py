# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import collections
from functools import partial
import itertools
import unittest

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

import jax
from jax import numpy as jnp

from jax._src import dtypes
from jax._src import test_util as jtu

from jax.config import config

config.parse_flags_with_absl()
FLAGS = config.FLAGS

numpy_version = jtu.numpy_version()

nonempty_nonscalar_array_shapes = [(4,), (3, 4), (3, 1), (1, 4), (2, 1, 4), (2, 3, 4)]
nonempty_array_shapes = [()] + nonempty_nonscalar_array_shapes
one_dim_array_shapes = [(1,), (6,), (12,)]
empty_array_shapes = [
    (0,),
    (0, 4),
    (3, 0),
]

scalar_shapes = [jtu.NUMPY_SCALAR_SHAPE, jtu.PYTHON_SCALAR_SHAPE]
array_shapes = nonempty_array_shapes + empty_array_shapes
nonzerodim_shapes = nonempty_nonscalar_array_shapes + empty_array_shapes
nonempty_shapes = scalar_shapes + nonempty_array_shapes
all_shapes = scalar_shapes + array_shapes

float_dtypes = jtu.dtypes.all_floating
complex_dtypes = jtu.dtypes.complex
int_dtypes = jtu.dtypes.all_integer
unsigned_dtypes = jtu.dtypes.all_unsigned
bool_dtypes = jtu.dtypes.boolean
default_dtypes = float_dtypes + int_dtypes
inexact_dtypes = float_dtypes + complex_dtypes
number_dtypes = float_dtypes + complex_dtypes + int_dtypes + unsigned_dtypes
all_dtypes = number_dtypes + bool_dtypes

python_scalar_dtypes = [jnp.bool_, jnp.int_, jnp.float_, jnp.complex_]


def _valid_dtypes_for_shape(shape, dtypes):
    # Not all (shape, dtype) pairs are valid. In particular, Python scalars only
    # have one type in each category (float, bool, etc.)
    if shape is jtu.PYTHON_SCALAR_SHAPE:
        return [t for t in dtypes if t in python_scalar_dtypes]
    return dtypes


def _shape_and_dtypes(shapes, dtypes):
    for shape in shapes:
        for dtype in _valid_dtypes_for_shape(shape, dtypes):
            yield (shape, dtype)


def _compatible_shapes(shape):
    if np.ndim(shape) == 0 or shape in scalar_shapes:
        return [shape]
    return (shape[n:] for n in range(len(shape) + 1))


def _get_y_shapes(y_dtype, shape, rowvar):
    # Helper function for testCov.
    if y_dtype is None:
        return [None]
    if len(shape) == 1:
        return [shape]
    elif rowvar or shape[0] == 1:
        return [(1, shape[-1]), (2, shape[-1]), (5, shape[-1])]
    return [(shape[0], 1), (shape[0], 2), (shape[0], 5)]


OpRecord = collections.namedtuple(
    "OpRecord",
    [
        "name",
        "nargs",
        "dtypes",
        "shapes",
        "rng_factory",
        "diff_modes",
        "test_name",
        "check_dtypes",
        "tolerance",
        "inexact",
        "kwargs",
    ],
)


def op_record(
    name,
    nargs,
    dtypes,
    shapes,
    rng_factory,
    diff_modes,
    test_name=None,
    check_dtypes=True,
    tolerance=None,
    inexact=False,
    kwargs=None,
):
    test_name = test_name or name
    return OpRecord(
        name,
        nargs,
        dtypes,
        shapes,
        rng_factory,
        diff_modes,
        test_name,
        check_dtypes,
        tolerance,
        inexact,
        kwargs,
    )


JAX_REDUCER_RECORDS = [
    op_record(
        "mean", 1, number_dtypes, nonempty_shapes, jtu.rand_default, [], inexact=True
    ),
    op_record("prod", 1, all_dtypes, all_shapes, jtu.rand_small_positive, []),
    op_record("sum", 1, all_dtypes, all_shapes, jtu.rand_default, []),
    op_record(
        "nanmean",
        1,
        inexact_dtypes,
        nonempty_shapes,
        jtu.rand_some_nan,
        [],
        inexact=True,
    ),
    op_record("nanprod", 1, all_dtypes, all_shapes, jtu.rand_some_nan, []),
    op_record("nansum", 1, number_dtypes, all_shapes, jtu.rand_some_nan, []),
]

JAX_REDUCER_INITIAL_RECORDS = [
    op_record("prod", 1, all_dtypes, all_shapes, jtu.rand_small_positive, []),
    op_record(
        "sum",
        1,
        all_dtypes,
        all_shapes,
        jtu.rand_default,
        [],
        tolerance={jnp.bfloat16: 2e-2},
    ),
    op_record("max", 1, all_dtypes, all_shapes, jtu.rand_default, []),
    op_record("min", 1, all_dtypes, all_shapes, jtu.rand_default, []),
]
if numpy_version >= (1, 22):  # initial & where keywords added in numpy 1.22
    JAX_REDUCER_INITIAL_RECORDS += [
        op_record(
            "nanprod", 1, inexact_dtypes, all_shapes, jtu.rand_small_positive, []
        ),
        op_record(
            "nansum",
            1,
            inexact_dtypes,
            all_shapes,
            jtu.rand_default,
            [],
            tolerance={jnp.bfloat16: 3e-2},
        ),
        op_record("nanmax", 1, inexact_dtypes, all_shapes, jtu.rand_default, []),
        op_record("nanmin", 1, inexact_dtypes, all_shapes, jtu.rand_default, []),
    ]

JAX_REDUCER_WHERE_NO_INITIAL_RECORDS = [
    op_record("all", 1, bool_dtypes, all_shapes, jtu.rand_some_zero, []),
    op_record("any", 1, bool_dtypes, all_shapes, jtu.rand_some_zero, []),
    op_record(
        "mean", 1, all_dtypes, nonempty_shapes, jtu.rand_default, [], inexact=True
    ),
    op_record(
        "var", 1, all_dtypes, nonempty_shapes, jtu.rand_default, [], inexact=True
    ),
    op_record(
        "std", 1, all_dtypes, nonempty_shapes, jtu.rand_default, [], inexact=True
    ),
]
if numpy_version >= (1, 22):  # where keyword added in numpy 1.22
    JAX_REDUCER_WHERE_NO_INITIAL_RECORDS += [
        op_record(
            "nanmean",
            1,
            inexact_dtypes,
            nonempty_shapes,
            jtu.rand_default,
            [],
            inexact=True,
            tolerance={np.float16: 3e-3},
        ),
        op_record(
            "nanvar",
            1,
            inexact_dtypes,
            nonempty_shapes,
            jtu.rand_default,
            [],
            inexact=True,
            tolerance={np.float16: 3e-3},
        ),
        op_record(
            "nanstd",
            1,
            inexact_dtypes,
            nonempty_shapes,
            jtu.rand_default,
            [],
            inexact=True,
            tolerance={np.float16: 1e-3},
        ),
    ]

JAX_REDUCER_NO_DTYPE_RECORDS = [
    op_record("all", 1, all_dtypes, all_shapes, jtu.rand_some_zero, []),
    op_record("any", 1, all_dtypes, all_shapes, jtu.rand_some_zero, []),
    op_record("max", 1, all_dtypes, nonempty_shapes, jtu.rand_default, []),
    op_record("min", 1, all_dtypes, nonempty_shapes, jtu.rand_default, []),
    op_record(
        "var",
        1,
        all_dtypes,
        nonempty_shapes,
        jtu.rand_default,
        [],
        inexact=True,
        tolerance={jnp.bfloat16: 2e-2},
    ),
    op_record(
        "std", 1, all_dtypes, nonempty_shapes, jtu.rand_default, [], inexact=True
    ),
    op_record("nanmax", 1, all_dtypes, nonempty_shapes, jtu.rand_some_nan, []),
    op_record("nanmin", 1, all_dtypes, nonempty_shapes, jtu.rand_some_nan, []),
    op_record(
        "nanvar", 1, all_dtypes, nonempty_shapes, jtu.rand_some_nan, [], inexact=True
    ),
    op_record(
        "nanstd", 1, all_dtypes, nonempty_shapes, jtu.rand_some_nan, [], inexact=True
    ),
    op_record("ptp", 1, number_dtypes, nonempty_shapes, jtu.rand_default, []),
]

JAX_REDUCER_PROMOTE_INT_RECORDS = [
    op_record("prod", 1, all_dtypes, all_shapes, jtu.rand_small_positive, []),
    op_record("sum", 1, all_dtypes, all_shapes, jtu.rand_default, []),
]


def _reducer_output_dtype(
    name: str, input_dtype: np.dtype, promote_integers: bool = True
) -> np.dtype:
    if name in ["sum", "prod", "nansum", "nanprod"]:
        if input_dtype == bool:
            input_dtype = dtypes.to_numeric_dtype(input_dtype)
        if promote_integers:
            if dtypes.issubdtype(input_dtype, np.integer):
                default_int = dtypes.canonicalize_dtype(
                    dtypes.uint
                    if dtypes.issubdtype(input_dtype, np.unsignedinteger)
                    else dtypes.int_
                )
                if np.iinfo(input_dtype).bits < np.iinfo(default_int).bits:
                    return default_int
    return input_dtype


class JaxNumpyReducerTests(jtu.JaxTestCase):
    """Tests for LAX-backed Numpy reduction operations."""

    @jtu.sample_product(
        [
            dict(shape=shape, dtype=dtype, axis=axis, weights_shape=weights_shape)
            for shape, dtype in _shape_and_dtypes(nonempty_shapes, number_dtypes)
            for axis in list(range(-len(shape), len(shape)))
            + [None]
            + [tuple(range(len(shape)))]
            # `weights_shape` is either `None`, same as the averaged axis, or same as
            # that of the input
            for weights_shape in (
                [None, shape]
                if axis is None or len(shape) == 1 or isinstance(axis, tuple)
                else [None, (shape[axis],), shape]
            )
        ],
        keepdims=([False, True] if numpy_version >= (1, 23) else [None]),
        returned=[False, True],
    )
    def testAverage(self, shape, dtype, axis, weights_shape, returned, keepdims):
        rng = jtu.rand_default(self.rng())
        kwds = dict(returned=returned)
        if keepdims is not None:
            kwds["keepdims"] = keepdims
        if weights_shape is None:
            np_fun = lambda x: np.average(x, axis, **kwds)
            jnp_fun = lambda x: jnp.average(x, axis, **kwds)
            args_maker = lambda: [rng(shape, dtype)]
        else:
            np_fun = lambda x, weights: np.average(x, axis, weights, **kwds)
            jnp_fun = lambda x, weights: jnp.average(x, axis, weights, **kwds)
            args_maker = lambda: [rng(shape, dtype), rng(weights_shape, dtype)]
        np_fun = jtu.promote_like_jnp(np_fun, inexact=True)
        tol = {
            dtypes.bfloat16: 2e-1,
            np.float16: 1e-2,
            np.float32: 1e-5,
            np.float64: 1e-12,
            np.complex64: 1e-5,
        }
        check_dtypes = shape is not jtu.PYTHON_SCALAR_SHAPE and numpy_version >= (1, 22)
        if (
            numpy_version == (1, 23, 0)
            and keepdims
            and weights_shape is not None
            and axis is not None
        ):
            # Known failure: https://github.com/numpy/numpy/issues/21850
            pass
        else:
            try:
                self._CheckAgainstNumpy(
                    np_fun, jnp_fun, args_maker, check_dtypes=check_dtypes, tol=tol
                )
            except ZeroDivisionError:
                self.skipTest("don't support checking for ZeroDivisionError")
        self._CompileAndCheck(
            jnp_fun, args_maker, check_dtypes=check_dtypes, rtol=tol, atol=tol
        )
