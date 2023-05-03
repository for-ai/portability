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
import itertools
import unittest
from functools import partial

import jax
import numpy as np
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax._src import dtypes
from jax._src import test_util as jtu
from jax.config import config

from ..utils.timer_wrapper import jax_op_timer, partial_timed

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
    op_record("prod", 1, all_dtypes, all_shapes, jtu.rand_small_positive, []),
]

JAX_REDUCER_INITIAL_RECORDS = []

if numpy_version >= (1, 22):  # initial & where keywords added in numpy 1.22
    JAX_REDUCER_INITIAL_RECORDS += []

JAX_REDUCER_WHERE_NO_INITIAL_RECORDS = []
if numpy_version >= (1, 22):  # where keyword added in numpy 1.22
    JAX_REDUCER_WHERE_NO_INITIAL_RECORDS += []

JAX_REDUCER_NO_DTYPE_RECORDS = []

JAX_REDUCER_PROMOTE_INT_RECORDS = []


def _reducer_output_dtype(
    name: str, input_dtype: np.dtype, promote_integers: bool = True
) -> np.dtype:
    if name in ["prod"]:
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

    def _GetArgsMaker(self, rng, shapes, dtypes, np_arrays=True):
        def f():
            out = [
                rng(shape, dtype or jnp.float_) for shape, dtype in zip(shapes, dtypes)
            ]
            if np_arrays:
                return out
            return [
                jnp.asarray(a) if isinstance(a, (np.ndarray, np.generic)) else a
                for a in out
            ]

        return f

    @parameterized.parameters(
        itertools.chain.from_iterable(
            jtu.sample_product_testcases(
                [dict(name=rec.name, rng_factory=rec.rng_factory, inexact=rec.inexact)],
                [
                    dict(shape=shape, axis=axis, dtype=dtype)
                    for shape in rec.shapes
                    for dtype in rec.dtypes
                    for axis in list(range(-len(shape), len(shape))) + [None]
                    if jtu.is_valid_shape(shape, dtype)
                ],
                out_dtype=[
                    out_dtype
                    for out_dtype in [None] + rec.dtypes
                    if out_dtype not in unsigned_dtypes
                ],
                keepdims=[False, True],
            )
            for rec in JAX_REDUCER_RECORDS
        )
    )
    def testReducer(
        self, name, rng_factory, shape, dtype, out_dtype, axis, keepdims, inexact
    ):
        np_op = getattr(np, name)
        jnp_op = getattr(jnp, name)
        rng = rng_factory(self.rng())

        @jtu.ignore_warning(category=np.ComplexWarning)
        @jtu.ignore_warning(category=RuntimeWarning, message="mean of empty slice.*")
        @jtu.ignore_warning(category=RuntimeWarning, message="overflow encountered.*")
        def np_fun(x):
            x = np.asarray(x)
            if inexact:
                x = x.astype(dtypes.to_inexact_dtype(x.dtype))
            x_cast = x if dtype != jnp.bfloat16 else x.astype(np.float32)
            t = out_dtype if out_dtype != jnp.bfloat16 else np.float32
            if t is None:
                t = _reducer_output_dtype(name, x_cast.dtype)
            return np_op(x_cast, axis, dtype=t, keepdims=keepdims)
        timer = jax_op_timer()
        with timer:
            jnp_fun = lambda x: jnp_op(x, axis, dtype=out_dtype, keepdims=keepdims)
            timer.gen.send(jnp_fun)
        jnp_fun = jtu.ignore_warning(category=jnp.ComplexWarning)(jnp_fun)
        args_maker = lambda: [rng(shape, dtype)]
        tol_spec = {
            np.float16: 1e-2,
            np.int16: 2e-7,
            np.int32: 1e-3,
            np.float32: 1e-3,
            np.complex64: 1e-3,
            np.float64: 1e-5,
            np.complex128: 1e-5,
        }
        tol = jtu.tolerance(dtype, tol_spec)
        tol = max(tol, jtu.tolerance(out_dtype, tol_spec)) if out_dtype else tol
        self._CheckAgainstNumpy(
            np_fun,
            jnp_fun,
            args_maker,
            check_dtypes=jnp.bfloat16 not in (dtype, out_dtype),
            tol=tol,
        )
        self._CompileAndCheck(jnp_fun, args_maker, atol=tol, rtol=tol)
