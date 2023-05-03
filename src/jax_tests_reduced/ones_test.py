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
import copy
import inspect
import io
import itertools
import math
import unittest
import warnings
from functools import partial
from typing import Iterator, List, Optional, Tuple, cast
from unittest import SkipTest

import numpy as np
from absl.testing import absltest, parameterized

try:
    import numpy_dispatch
except ImportError:
    numpy_dispatch = None

import jax
import jax.ops
from jax import lax
from jax import numpy as jnp
from jax import tree_util
from jax._src import array, core, dtypes
from jax._src import test_util as jtu
from jax._src.lax import lax as lax_internal
from jax._src.numpy.util import ParsedDoc, _parse_numpydoc, _wraps
from jax._src.util import safe_zip
from jax.config import config
from jax.test_util import check_grads

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

from ..utils.timer_wrapper import jax_op_timer, partial_timed

python_scalar_dtypes = [jnp.bool_, jnp.int_, jnp.float_, jnp.complex_]

# uint64 is problematic because with any uint type it promotes to float:
int_dtypes_no_uint64 = [d for d in int_dtypes + unsigned_dtypes if d != np.uint64]


def _indexer_with_default_outputs(indexer, use_defaults=True):
    """Like jtu.with_jax_dtype_defaults, but for __getitem__ APIs"""

    class Indexer:
        @partial(jtu.with_jax_dtype_defaults, use_defaults=use_defaults)
        def __getitem__(self, *args):
            return indexer.__getitem__(*args)

    return Indexer()


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


JAX_ARGMINMAX_RECORDS = [
    op_record("argmin", 1, default_dtypes, nonempty_shapes, jtu.rand_some_equal, []),
    op_record("argmax", 1, default_dtypes, nonempty_shapes, jtu.rand_some_equal, []),
    op_record("nanargmin", 1, default_dtypes, nonempty_shapes, jtu.rand_some_nan, []),
    op_record("nanargmax", 1, default_dtypes, nonempty_shapes, jtu.rand_some_nan, []),
]


def _shapes_are_broadcast_compatible(shapes):
    try:
        lax.broadcast_shapes(*(() if s in scalar_shapes else s for s in shapes))
    except ValueError:
        return False
    else:
        return True


def _shapes_are_equal_length(shapes):
    return all(len(shape) == len(shapes[0]) for shape in shapes[1:])


class LaxBackedNumpyTests(jtu.JaxTestCase):
    """Tests for LAX-backed Numpy implementation."""

    @jtu.sample_product(
        op=["ones"],
        shape=[
            2,
            (),
            (2,),
            (3, 0),
            np.array((4, 5, 6), dtype=np.int32),
            np.array(4, dtype=np.int32),
        ],
        dtype=all_dtypes,
    )
    def testZerosOnes(self, op, shape, dtype):
        np_op = getattr(np, op)
        jnp_op = getattr(jnp, op)
        args_maker = lambda: []
        np_op = partial(np_op, shape, dtype)
        jnp_op = partial_timed(jnp_op, shape, dtype)
        self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
        self._CompileAndCheck(jnp_op, args_maker)
