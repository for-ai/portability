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
from ..utils.timer_wrapper import jax_op_timer, partial_timed

config.parse_flags_with_absl()
FLAGS = config.FLAGS


python_version = (sys.version_info[0], sys.version_info[1])
numpy_version = jtu.numpy_version()


def _check_instance(self, x):
    self.assertIsInstance(x, array.ArrayImpl)


class APITest(jtu.JaxTestCase):
    def test_dunder_jax_array(self):
        # https://github.com/google/jax/pull/4725

        class AlexArray:
            def __init__(self, jax_val):
                self.jax_val = jax_val

            def __jax_array__(self):
                return self.jax_val

            dtype = property(lambda self: self.jax_val.dtype)
            shape = property(lambda self: self.jax_val.shape)

        x = AlexArray(jnp.array([1.0, 2.0, 3.0]))
        y = jnp.sin(x)
        self.assertAllClose(y, jnp.sin(jnp.array([1.0, 2.0, 3.0])))
        y = api.grad(api.jit(lambda x: jnp.sin(x).sum()))(x)
        self.assertAllClose(y, jnp.cos(jnp.array([1.0, 2.0, 3.0])))

        x = jnp.array(1)
        a = AlexArray(x)
        op = partial_timed(jnp.isscalar)
        for f in [op]:
            self.assertEqual(f(x), f(a))

        x = AlexArray(jnp.array(1))
        a1 = jnp.array(x)
        self.assertAllClose(1, a1)

        a2 = jnp.array(((x, x), [x, x]))
        self.assertAllClose(np.array(((1, 1), (1, 1))), a2)
