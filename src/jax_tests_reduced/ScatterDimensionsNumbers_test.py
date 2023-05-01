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
from __future__ import annotations

from functools import partial
import itertools
import math
import operator
import types
import unittest
from unittest import SkipTest
from typing import Tuple

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

import jax
from jax._src import core
from jax import lax
import jax.numpy as jnp
from jax.test_util import check_grads
from jax import tree_util
import jax.util

from jax.interpreters import xla
from jax._src.interpreters import mlir
from jax.interpreters import batching
from jax._src import array
from jax._src.lib.mlir.dialects import hlo
from jax._src import dtypes
from jax._src.interpreters import pxla
from jax._src import test_util as jtu
from jax._src import lax_reference
from jax._src.lax import lax as lax_internal
from jax._src.internal_test_util import lax_test_util

from jax.config import config

config.parse_flags_with_absl()


### lax tests

# We check cases where the preferred type is at least as wide as the input
# type and where both are either both floating-point or both integral,
# which are the only supported configurations.
preferred_type_combinations = [
    (np.float16, np.float16),
    (np.float16, np.float32),
    (np.float16, np.float64),
    (dtypes.bfloat16, dtypes.bfloat16),
    (dtypes.bfloat16, np.float32),
    (dtypes.bfloat16, np.float64),
    (np.float32, np.float32),
    (np.float32, np.float64),
    (np.float64, np.float64),
    (np.int8, np.int8),
    (np.int8, np.int16),
    (np.int8, np.int32),
    (np.int8, np.int64),
    (np.int16, np.int16),
    (np.int16, np.int32),
    (np.int16, np.int64),
    (np.int32, np.int32),
    (np.int32, np.int64),
    (np.int64, np.int64),
    (np.complex64, np.complex64),
    (np.complex64, np.complex128),
    (np.complex128, np.complex128),
    (np.int8, np.float16),
    (np.int8, dtypes.bfloat16),
    (np.int8, np.float32),
    (np.int8, np.float64),
    (np.int16, np.float16),
    (np.int16, dtypes.bfloat16),
    (np.int16, np.float32),
    (np.int16, np.float64),
    (np.int32, np.float32),
    (np.int32, np.float64),
    (np.int64, np.float64),
]


class LaxTest(jtu.JaxTestCase):
    @parameterized.named_parameters(
        {
            "testcase_name": f"_{testcase_name}",
            "operand_shape": operand_shape,
            "indices_shape": indices_shape,
            "dimension_numbers": lax.GatherDimensionNumbers(
                offset_dims=offset_dims,
                collapsed_slice_dims=collapsed_slice_dims,
                start_index_map=start_index_map,
            ),
            "slice_sizes": slice_sizes,
            "msg": msg,
        }
        for (
            testcase_name,
            operand_shape,
            indices_shape,
            offset_dims,
            collapsed_slice_dims,
            start_index_map,
            slice_sizes,
            msg,
        ) in [
            (
                "NonAscendingWindowIndices",
                (10, 9, 8, 7, 6),
                (5, 4, 3, 2, 1),
                (4, 5, 6, 8, 7),
                (),
                (0, 1, 2, 3, 4),
                (10, 9, 8, 7, 6),
                "offset_dims in gather op must be sorted",
            ),
            (
                "RepeatedWindowIndices",
                (10, 9, 8, 7, 6),
                (5, 4, 3, 2, 1),
                (4, 5, 6, 7, 7),
                (),
                (0, 1, 2, 3, 4),
                (10, 9, 8, 7, 6),
                "offset_dims in gather op must not repeat",
            ),
            (
                "WindowIndexOutOfBounds",
                (10, 9, 8, 7, 6),
                (5, 4, 3, 2, 1),
                (4, 5, 100, 101, 102),
                (),
                (0, 1, 2, 3, 4),
                (10, 9, 8, 7, 6),
                "Offset dimension 2 in gather op is out of bounds",
            ),
            (
                "WindowIndexBarelyOutOfBounds",
                (10, 9, 8, 7, 6),
                (5, 4, 3, 2, 1),
                (4, 5, 6, 7, 9),
                (),
                (0, 1, 2, 3, 4),
                (10, 9, 8, 7, 6),
                "Offset dimension 4 in gather op is out of bounds",
            ),
            (
                "MismatchingElidedWindowDims",
                (10, 9, 8, 7, 6),
                (5, 4, 3, 2, 5),
                (4, 5, 6, 7, 8),
                (4,),
                (0, 1, 2, 3, 4),
                (10, 9, 8, 7, 6),
                (
                    "All components of the offset index in a gather op must either be a "
                    "offset dimension or explicitly collapsed"
                ),
            ),
            (
                "OutOfBoundsWindowToInputMapping",
                (10, 9, 8, 7, 6),
                (5, 4, 3, 2, 5),
                (4, 5, 6, 7, 8),
                (0, 1, 2, 3, 19),
                (0, 1, 2, 3, 4),
                (10, 9, 8, 7, 6),
                "Invalid collapsed_slice_dims set in gather op; valid range is",
            ),
            (
                "RepeatedWindowToInputMapping",
                (10, 9, 8, 7, 6),
                (5, 4, 3, 2, 5),
                (4, 5, 6, 7, 8),
                (0, 1, 2, 3, 3),
                (0, 1, 2, 3, 4),
                (10, 9, 8, 7, 6),
                "collapsed_slice_dims in gather op must not repeat",
            ),
            (
                "MismatchingGatherToInputMapping",
                (10, 9, 8, 7, 6),
                (5, 4, 3, 2, 5),
                (4, 5, 6, 7, 8),
                (),
                (0, 1, 2, 3),
                (10, 9, 8, 7, 6),
                (
                    "Gather op has 4 elements in start_index_map and the bound of "
                    "dimension index_vector_dim=4 of indices is 5. These two "
                    "numbers must be equal."
                ),
            ),
            (
                "OutOfBoundsGatherToInputMapping",
                (10, 9, 8, 7, 6),
                (5, 4, 3, 2, 5),
                (4, 5, 6, 7, 8),
                (),
                (0, 1, 2, 3, 7),
                (10, 9, 8, 7, 6),
                "Invalid start_index_map",
            ),
            (
                "RepeatedGatherToInputMapping",
                (10, 9, 8, 7, 6),
                (5, 4, 3, 2, 5),
                (4, 5, 6, 7, 8),
                (),
                (0, 1, 2, 3, 3),
                (10, 9, 8, 7, 6),
                "start_index_map in gather op must not repeat",
            ),
            (
                "NonAscendingElidedWindowDims",
                (10, 9, 8, 7, 6),
                (5, 4, 3, 2, 5),
                (4, 5, 6, 7, 8),
                (2, 1),
                (0, 1, 2, 3, 4),
                (10, 9, 8, 7, 6),
                "collapsed_slice_dims in gather op must be sorted",
            ),
            (
                "WindowBoundsTooLarge",
                (10, 9, 8, 7, 6),
                (5, 4, 3, 2, 5),
                (4, 5, 6, 7),
                (2,),
                (0, 1, 2, 3, 4),
                (10, 9, 8, 100, 6),
                "Slice size at index 3 in gather op is out of range",
            ),
            (
                "MismatchingNumberOfWindowBounds",
                (10, 9, 8, 7, 6),
                (5, 4, 3, 2, 5),
                (4, 5, 6, 7),
                (),
                (0, 1, 2, 3, 4),
                (10, 9, 8, 7),
                "Gather op must have one slice size for every input dimension",
            ),
            (
                "WindowBoundsNot1ForElidedDim",
                (10, 9, 8, 7, 6),
                (5, 4, 3, 2, 5),
                (4, 5, 6, 7),
                (1,),
                (0, 1, 2, 3, 4),
                (10, 9, 8, 7, 6),
                (
                    "Gather op can only collapse slice dims with bound 1, but bound "
                    "is 9 for index 1 at position 0."
                ),
            ),
        ]
    )
    def testGatherShapeCheckingRule(
        self, operand_shape, indices_shape, dimension_numbers, slice_sizes, msg
    ):
        operand = np.ones(operand_shape, dtype=np.int32)
        indices = np.ones(indices_shape, dtype=np.int32)

        with self.assertRaisesRegex(TypeError, msg):
            lax.gather(operand, indices, dimension_numbers, slice_sizes)

    @jtu.sample_product(
        [
            dict(arg_shape=arg_shape, idxs=idxs, update_shape=update_shape, dnums=dnums)
            for arg_shape, idxs, update_shape, dnums in [
                (
                    (5,),
                    np.array([[0], [2]]),
                    (2,),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(),
                        inserted_window_dims=(0,),
                        scatter_dims_to_operand_dims=(0,),
                    ),
                ),
                (
                    (10,),
                    np.array([[0], [0], [0]]),
                    (3, 2),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(1,),
                        inserted_window_dims=(),
                        scatter_dims_to_operand_dims=(0,),
                    ),
                ),
                (
                    (
                        10,
                        5,
                    ),
                    np.array([[0], [2], [1]]),
                    (3, 3),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(1,),
                        inserted_window_dims=(0,),
                        scatter_dims_to_operand_dims=(0,),
                    ),
                ),
            ]
        ],
        dtype=lax_test_util.inexact_dtypes,
        mode=["clip", "fill", None],
    )
    def testScatterAdd(self, arg_shape, dtype, idxs, update_shape, dnums, mode):
        rng = jtu.rand_default(self.rng())
        rng_idx = jtu.rand_int(self.rng(), high=max(arg_shape))
        rand_idxs = lambda: rng_idx(idxs.shape, idxs.dtype)
        args_maker = lambda: [
            rng(arg_shape, dtype),
            rand_idxs(),
            rng(update_shape, dtype),
        ]
        fun = partial(lax.scatter_add, dimension_numbers=dnums, mode=mode)
        self._CompileAndCheck(fun, args_maker)

    @jtu.sample_product(
        [
            dict(arg_shape=arg_shape, idxs=idxs, update_shape=update_shape, dnums=dnums)
            for arg_shape, idxs, update_shape, dnums in [
                (
                    (5,),
                    np.array([[0], [2]]),
                    (2,),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(),
                        inserted_window_dims=(0,),
                        scatter_dims_to_operand_dims=(0,),
                    ),
                ),
                (
                    (10,),
                    np.array([[0], [0], [0]]),
                    (3, 2),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(1,),
                        inserted_window_dims=(),
                        scatter_dims_to_operand_dims=(0,),
                    ),
                ),
                (
                    (
                        10,
                        5,
                    ),
                    np.array([[0], [2], [1]], dtype=np.uint64),
                    (3, 3),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(1,),
                        inserted_window_dims=(0,),
                        scatter_dims_to_operand_dims=(0,),
                    ),
                ),
            ]
        ],
        dtype=lax_test_util.float_dtypes,
    )
    def testScatterMin(self, arg_shape, dtype, idxs, update_shape, dnums):
        rng = jtu.rand_default(self.rng())
        rng_idx = jtu.rand_int(self.rng(), high=max(arg_shape))
        rand_idxs = lambda: rng_idx(idxs.shape, idxs.dtype)
        args_maker = lambda: [
            rng(arg_shape, dtype),
            rand_idxs(),
            rng(update_shape, dtype),
        ]
        fun = partial(lax.scatter_min, dimension_numbers=dnums)
        self._CompileAndCheck(fun, args_maker)

    @jtu.sample_product(
        [
            dict(arg_shape=arg_shape, idxs=idxs, update_shape=update_shape, dnums=dnums)
            for arg_shape, idxs, update_shape, dnums in [
                (
                    (5,),
                    np.array([[0], [2]]),
                    (2,),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(),
                        inserted_window_dims=(0,),
                        scatter_dims_to_operand_dims=(0,),
                    ),
                ),
                (
                    (10,),
                    np.array([[0], [0], [0]]),
                    (3, 2),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(1,),
                        inserted_window_dims=(),
                        scatter_dims_to_operand_dims=(0,),
                    ),
                ),
                (
                    (
                        10,
                        5,
                    ),
                    np.array([[0], [2], [1]]),
                    (3, 3),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(1,),
                        inserted_window_dims=(0,),
                        scatter_dims_to_operand_dims=(0,),
                    ),
                ),
            ]
        ],
        dtype=lax_test_util.float_dtypes,
    )
    def testScatterMax(self, arg_shape, dtype, idxs, update_shape, dnums):
        rng = jtu.rand_default(self.rng())
        rng_idx = jtu.rand_int(self.rng(), high=max(arg_shape))
        rand_idxs = lambda: rng_idx(idxs.shape, idxs.dtype)
        args_maker = lambda: [
            rng(arg_shape, dtype),
            rand_idxs(),
            rng(update_shape, dtype),
        ]
        fun = partial(lax.scatter_max, dimension_numbers=dnums)
        self._CompileAndCheck(fun, args_maker)

    @jtu.sample_product(
        [
            dict(arg_shape=arg_shape, idxs=idxs, update_shape=update_shape, dnums=dnums)
            for arg_shape, idxs, update_shape, dnums in [
                (
                    (5,),
                    np.array([[0], [2]]),
                    (2,),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(),
                        inserted_window_dims=(0,),
                        scatter_dims_to_operand_dims=(0,),
                    ),
                ),
                (
                    (10,),
                    np.array([[0], [0], [0]]),
                    (3, 2),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(1,),
                        inserted_window_dims=(),
                        scatter_dims_to_operand_dims=(0,),
                    ),
                ),
                (
                    (
                        10,
                        5,
                    ),
                    np.array([[0], [2], [1]]),
                    (3, 3),
                    lax.ScatterDimensionNumbers(
                        update_window_dims=(1,),
                        inserted_window_dims=(0,),
                        scatter_dims_to_operand_dims=(0,),
                    ),
                ),
            ]
        ],
        dtype=lax_test_util.float_dtypes,
    )
    def testScatter(self, arg_shape, dtype, idxs, update_shape, dnums):
        rng = jtu.rand_default(self.rng())
        rng_idx = jtu.rand_int(self.rng(), high=max(arg_shape))
        rand_idxs = lambda: rng_idx(idxs.shape, idxs.dtype)
        args_maker = lambda: [
            rng(arg_shape, dtype),
            rand_idxs(),
            rng(update_shape, dtype),
        ]
        fun = partial(lax.scatter, dimension_numbers=dnums)
        self._CompileAndCheck(fun, args_maker)

    # These tests are adapted from the corresponding tests in
    # tensorflow/compiler/xla/service/shape_inference_test.cc with slight
    # variations to account for the implicit setting of index_vector_dim in JAX.
    @parameterized.named_parameters(
        {
            "testcase_name": f"_{testcase_name}",
            "operand_shape": operand_shape,
            "indices": indices,
            "update_shape": update_shape,
            "dimension_numbers": lax.ScatterDimensionNumbers(
                update_window_dims=update_window_dims,
                inserted_window_dims=inserted_window_dims,
                scatter_dims_to_operand_dims=scatter_dims_to_operand_dims,
            ),
            "msg": msg,
        }
        for (
            testcase_name,
            operand_shape,
            indices,
            update_shape,
            update_window_dims,
            inserted_window_dims,
            scatter_dims_to_operand_dims,
            msg,
        ) in [
            (
                "ScatterWithUpdatesBiggerThanInput",
                (64, 48),
                np.zeros((32, 1)),
                (65, 32),
                (0,),
                (1,),
                (1,),
                "Bounds of the window dimensions",
            ),
            (
                "ScatterWithUpdatesBiggerThanInputV2",
                (64, 48),
                np.zeros((32, 1)),
                (32, 49),
                (1,),
                (0,),
                (1,),
                "Bounds of the window dimensions",
            ),
            (
                "ScatterWithUpdatesNotMatchingIndices",
                (64, 48),
                np.zeros((32, 1)),
                (64, 31),
                (0,),
                (1,),
                (1,),
                "Bounds of the scatter dimensions",
            ),
            (
                "ScatterWithUpdatesNotMatchingIndicesV2",
                (64, 48),
                np.zeros((32, 1)),
                (31, 48),
                (1,),
                (0,),
                (1,),
                "Bounds of the scatter dimensions",
            ),
            (
                "ScatterNdWithUpdatesBiggerThanInput",
                (64, 48),
                np.zeros((10, 9, 8, 7, 1)),
                (10, 9, 8, 7, 65),
                (4,),
                (1,),
                (0,),
                "Bounds of the window dimensions",
            ),
            (
                "ScatterNdWithUpdatesNotMatchingIndices",
                (64, 48),
                np.zeros((10, 9, 8, 7, 1)),
                (9, 9, 8, 7, 64),
                (4,),
                (1,),
                (0,),
                "Bounds of the scatter dimensions",
            ),
            (
                "InvalidUpdates",
                (50, 49, 48, 47, 46),
                np.zeros((10, 9, 8, 7, 5)),
                (10, 9, 8, 7, 3, 2, 4, 1),
                (4, 5, 6),
                (1, 2),
                (0, 1, 2, 3, 4),
                "Updates tensor must be of rank 7; got 8.",
            ),
            (
                "NonAscendingUpdateWindowDims",
                (6, 5, 4, 3, 2),
                np.zeros((5, 4, 3, 2, 1)),
                (10, 9, 8, 7, 6, 5, 4, 3, 2),
                (4, 5, 6, 8, 7),
                (),
                (0, 1, 2, 3, 4),
                "update_window_dims in scatter op must be sorted",
            ),
            (
                "RepeatedUpdateWindowDims",
                (6, 5, 4, 3, 2),
                np.zeros((5, 4, 3, 2, 1)),
                (10, 9, 8, 7, 6, 5, 4, 3, 2),
                (4, 5, 6, 7, 7),
                (),
                (0, 1, 2, 3, 4),
                "update_window_dims in scatter op must not repeat",
            ),
            (
                "OutOfBoundsUpdateWindowDims",
                (6, 5, 4, 3, 2),
                np.zeros((5, 4, 3, 2, 1)),
                (10, 9, 8, 7, 6, 5, 4, 3, 2),
                (4, 5, 6, 7, 9),
                (),
                (0, 1, 2, 3, 4),
                "Invalid update_window_dims set in scatter op",
            ),
            (
                "NonAscendingInsertedWindowDims",
                (50, 49, 48, 47, 46),
                np.zeros((10, 9, 8, 7, 5)),
                (10, 9, 8, 7, 3, 2, 4),
                (4, 5, 6),
                (2, 1),
                (0, 1, 2, 3, 4),
                "inserted_window_dims in scatter op must be sorted",
            ),
            (
                "RepeatedInsertedWindowDims",
                (50, 49, 48, 47, 46),
                np.zeros((10, 9, 8, 7, 5)),
                (10, 9, 8, 7, 3, 2, 4),
                (4, 5, 6),
                (1, 1),
                (0, 1, 2, 3, 4),
                "inserted_window_dims in scatter op must not repeat",
            ),
            (
                "OutOfBoundsInsertedWindowDims",
                (50, 49, 48, 47, 46),
                np.zeros((10, 9, 8, 7, 5)),
                (10, 9, 8, 7, 3, 2, 4),
                (4, 5, 6),
                (1, 5),
                (0, 1, 2, 3, 4),
                "Invalid inserted_window_dims set in scatter op",
            ),
            (
                "MismatchingScatterDimsToOperandDims",
                (50, 49, 48, 47, 46),
                np.zeros((10, 9, 8, 7, 5)),
                (10, 9, 8, 7, 3, 2, 4),
                (4, 5, 6),
                (1, 2),
                (0, 1, 2, 3),
                (
                    "Scatter op has 4 elements in scatter_dims_to_operand_dims and "
                    "the bound of dimension index_vector_dim=4 of indices "
                    "is 5. These two numbers must be equal"
                ),
            ),
            (
                "OutOfBoundsScatterDimsToOperandDims",
                (50, 49, 48, 47, 46),
                np.zeros((10, 9, 8, 7, 5)),
                (10, 9, 8, 7, 3, 2, 4),
                (4, 5, 6),
                (1, 2),
                (0, 1, 2, 3, 10),
                "Invalid scatter_dims_to_operand_dims mapping",
            ),
            (
                "RepeatedValuesInScatterDimsToOperandDims",
                (50, 49, 48, 47, 46),
                np.zeros((10, 9, 8, 7, 5)),
                (10, 9, 8, 7, 3, 2, 4),
                (4, 5, 6),
                (1, 2),
                (0, 1, 2, 2, 3),
                "scatter_dims_to_operand_dims in scatter op must not repeat",
            ),
            (
                "InsufficientWindowDims",
                (50, 49, 48, 47, 46),
                np.zeros((10, 9, 8, 7, 5)),
                (10, 9, 8, 7, 3, 2, 4),
                (4, 5, 6),
                (1,),
                (0, 1, 2, 3),
                (
                    "Scatter op has window of size 4; doesn't match operand of "
                    "rank 5."
                ),
            ),
        ]
    )
    def testScatterShapeCheckingRule(
        self, operand_shape, indices, update_shape, dimension_numbers, msg
    ):
        def f(x, y):
            operand = lax.broadcast(x, operand_shape)
            updates = lax.broadcast(y, update_shape)
            return lax.scatter(operand, indices, updates, dimension_numbers)

        with self.assertRaisesRegex(TypeError, msg):
            jax.eval_shape(f, np.int32(1), np.int32(1))
