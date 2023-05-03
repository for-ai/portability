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
import concurrent.futures
import copy
import enum
import functools
import gc
import importlib
import inspect
import itertools as it
import operator
import operator as op
import os
import platform
import re
import subprocess
import sys
import types
import unittest
import warnings
import weakref
from contextlib import contextmanager
from functools import partial
from typing import Callable, List, NamedTuple, Optional

import jax
import jax._src.util as jax_util
import jax.numpy as jnp
import numpy as np
from absl import logging
from absl.testing import absltest, parameterized
from jax import custom_batching
from jax import custom_derivatives as custom_derivatives_public
from jax import (device_put, float0, grad, hessian, jacfwd, jacrev, jit, lax,
                 tree_util)
from jax._src import (api, api_util, array, core, custom_derivatives,
                      device_array, dtypes, lib)
from jax._src import linear_util as lu
from jax._src import prng
from jax._src import test_util as jtu
from jax._src import xla_bridge
from jax._src.ad_checkpoint import saved_residuals
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.lib import xla_client
from jax.ad_checkpoint import checkpoint as new_checkpoint
from jax.ad_checkpoint import checkpoint_name
from jax.config import config
from jax.errors import UnexpectedTracerError
from jax.experimental import pjit
from jax.interpreters import ad, batching, xla
from jax.sharding import PartitionSpec as P

from ..utils.timer_wrapper import jax_op_timer, partial_timed

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
        timer = jax_op_timer()
        with timer:
            jaxpr = api.make_jaxpr(f_)(x)
            timer.gen.send(jaxpr)
        self.assertIn("custom_transpose_call", str(jaxpr))

        timer = jax_op_timer()
        with timer:
            jaxpr_t = api.make_jaxpr(f_t)(x)
            timer.gen.send(jaxpr_t)
        self.assertNotIn("custom_transpose_call", str(jaxpr_t))
