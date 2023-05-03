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


JAX_REDUCER_INITIAL_RECORDS = [
    op_record("max", 1, all_dtypes, all_shapes, jtu.rand_default, []),
]


class JaxNumpyReducerTests(jtu.JaxTestCase):
    """Tests for LAX-backed Numpy reduction operations."""

    @jtu.sample_product(rec=JAX_REDUCER_INITIAL_RECORDS)
    def testReducerBadInitial(self, rec):
        jnp_op = getattr(jnp, rec.name)
        arr = jnp.ones((2, 3, 4))
        initial = jnp.zeros((1, 2, 3))
        msg = r"initial value must be a scalar. Got array of shape \(1, 2, 3\)"
        with self.assertRaisesRegex(ValueError, msg):
            jnp_op(arr, axis=-1, initial=initial)

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
                initial=[0, 1],
                keepdims=[False, True],
            )
            for rec in JAX_REDUCER_INITIAL_RECORDS
        )
    )
    def testReducerInitial(
        self, name, rng_factory, shape, dtype, axis, keepdims, initial, inexact
    ):
        np_op = getattr(np, name)
        timer = jax_op_timer()
        with timer:
            jnp_op = getattr(jnp, name)
            timer.gen.send(jnp_op)
        rng = rng_factory(self.rng())
        is_bf16_nan_test = (
            dtype == jnp.bfloat16 and rng_factory.__name__ == "rand_some_nan"
        )

        @jtu.ignore_warning(
            category=RuntimeWarning, message="Degrees of freedom <= 0 for slice.*"
        )
        @jtu.ignore_warning(category=np.ComplexWarning)
        def np_fun(x):
            x = np.asarray(x)
            if inexact:
                x = x.astype(dtypes.to_inexact_dtype(x.dtype))
            x_cast = x if not is_bf16_nan_test else x.astype(np.float32)
            res = np_op(x_cast, axis, keepdims=keepdims, initial=initial)
            res = res if not is_bf16_nan_test else res.astype(jnp.bfloat16)
            return res.astype(_reducer_output_dtype(name, x.dtype))
        timer = jax_op_timer()
        with timer:
            jnp_fun = lambda x: jnp_op(x, axis, keepdims=keepdims, initial=initial)
            timer.gen.send(jnp_fun)

        jnp_fun = jtu.ignore_warning(category=jnp.ComplexWarning)(jnp_fun)
        args_maker = lambda: [rng(shape, dtype)]
        tol = {jnp.bfloat16: 3e-2}
        self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, rtol=tol, atol=tol)
        self._CompileAndCheck(jnp_fun, args_maker)
