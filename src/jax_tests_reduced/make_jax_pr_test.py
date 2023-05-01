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
import collections.abc
from contextlib import contextmanager
import copy
import enum
from functools import partial
import inspect
import importlib
import operator
import os
import platform
import re
import subprocess
import sys
import types
from typing import Callable, List, Optional, NamedTuple
import unittest
import warnings
import weakref
import functools
import itertools as it
import operator as op
import gc

from absl import logging
from absl.testing import absltest, parameterized
import numpy as np

import concurrent.futures

import jax
import jax.numpy as jnp
from jax import float0, jit, grad, device_put, jacfwd, jacrev, hessian
from jax._src import core
from jax import lax
from jax import custom_batching
from jax._src import api, dtypes, lib, api_util
from jax.errors import UnexpectedTracerError
from jax.interpreters import ad
from jax._src.interpreters import mlir
from jax.interpreters import xla
from jax.interpreters import batching
from jax._src.interpreters import partial_eval as pe
from jax.sharding import PartitionSpec as P
from jax._src import array
from jax.experimental import pjit
from jax._src import custom_derivatives
from jax import custom_derivatives as custom_derivatives_public
from jax._src import device_array
from jax._src import prng
from jax._src import xla_bridge
from jax._src.lib import xla_client
from jax._src import test_util as jtu
from jax import tree_util
from jax._src import linear_util as lu
import jax._src.util as jax_util
from jax._src.ad_checkpoint import saved_residuals
from jax.ad_checkpoint import checkpoint as new_checkpoint, checkpoint_name

from jax.config import config

config.parse_flags_with_absl()
FLAGS = config.FLAGS


python_version = (sys.version_info[0], sys.version_info[1])
numpy_version = jtu.numpy_version()


def _check_instance(self, x):
    self.assertIsInstance(x, array.ArrayImpl)


def transpose_unary(f, x_example):
    def transposed(y):
        (x,) = api.linear_transpose(f, x_example)(y)
        return x

    return transposed


class _custom_transpose:
    def __init__(self, out_types, fun):
        self.out_types = out_types
        self.fun = api.custom_transpose(fun)

    def __getattr__(self, name):
        return getattr(self.fun, name)

    def __call__(self, *args):
        return self.fun(self.out_types, *args)


def custom_transpose(example_out):
    if isinstance(example_out, Callable):
        out_type = core.get_aval(0.0).at_least_vspace()
        return _custom_transpose(out_type, example_out)
    return partial(
        _custom_transpose,
        tree_util.tree_map(lambda x: core.get_aval(x).at_least_vspace(), example_out),
    )


class CustomTransposeTest(jtu.JaxTestCase):
    def test_make_jaxpr(self):
        def f(x, y):
            @custom_transpose(jnp.ones(2))
            def fn(r, x):
                return x / r

            @fn.def_transpose
            def tp(r, t):
                return 2 * t / r

            return x + fn(y, x)

        x = jnp.ones(2) * 6.0
        y = jnp.ones(2) * 3.0
        f_ = lambda x: f(x, y)
        f_t = transpose_unary(f_, x)

        jaxpr = api.make_jaxpr(f_)(x)
        self.assertIn("custom_transpose_call", str(jaxpr))

        jaxpr_t = api.make_jaxpr(f_t)(x)
        self.assertNotIn("custom_transpose_call", str(jaxpr_t))
