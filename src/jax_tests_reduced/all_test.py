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


JAX_REDUCER_WHERE_NO_INITIAL_RECORDS = [
    op_record("all", 1, bool_dtypes, all_shapes, jtu.rand_some_zero, []),
]


class JaxNumpyReducerTests(jtu.JaxTestCase):
    """Tests for LAX-backed Numpy reduction operations."""

    @parameterized.parameters(
        itertools.chain.from_iterable(
            jtu.sample_product_testcases(
                [
                    dict(
                        name=rec.name,
                        rng_factory=rec.rng_factory,
                        inexact=rec.inexact,
                        tol=rec.tolerance,
                    )
                ],
                [
                    dict(shape=shape, axis=axis, dtype=dtype, whereshape=whereshape)
                    for shape in rec.shapes
                    for dtype in rec.dtypes
                    for whereshape in _compatible_shapes(shape)
                    for axis in list(range(-len(shape), len(shape))) + [None]
                    if jtu.is_valid_shape(shape, dtype)
                ],
                keepdims=[False, True],
            )
            for rec in JAX_REDUCER_WHERE_NO_INITIAL_RECORDS
        )
    )
    def testReducerWhereNoInitial(
        self, name, rng_factory, shape, dtype, axis, keepdims, inexact, whereshape, tol
    ):
        np_op = getattr(np, name)
        jnp_op = getattr(jnp, name)
        rng = rng_factory(self.rng())
        is_bf16_nan_test = dtype == jnp.bfloat16
        # Do not pass where via args_maker as that is incompatible with _promote_like_jnp.
        where = jtu.rand_bool(self.rng())(whereshape, np.bool_)

        @jtu.ignore_warning(
            category=RuntimeWarning, message="Degrees of freedom <= 0 for slice.*"
        )
        @jtu.ignore_warning(category=RuntimeWarning, message="Mean of empty slice.*")
        @jtu.ignore_warning(
            category=RuntimeWarning, message="invalid value encountered.*"
        )
        @jtu.ignore_warning(category=np.ComplexWarning)
        def np_fun(x):
            x = np.asarray(x)
            if inexact:
                x = x.astype(dtypes.to_inexact_dtype(x.dtype))
            x_cast = x if not is_bf16_nan_test else x.astype(np.float32)
            res = np_op(x_cast, axis, keepdims=keepdims, where=where)
            res = res if not is_bf16_nan_test else res.astype(jnp.bfloat16)
            return res

        jnp_fun = lambda x: jnp_op(x, axis, keepdims=keepdims, where=where)
        jnp_fun = jtu.ignore_warning(category=jnp.ComplexWarning)(jnp_fun)
        args_maker = lambda: [rng(shape, dtype)]
        self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, atol=tol, rtol=tol)
        self._CompileAndCheck(jnp_fun, args_maker)
