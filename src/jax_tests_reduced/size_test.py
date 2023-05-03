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
        for f in [jnp.size]:
            timer = jax_op_timer()
            with timer:
                test = f(x)
                timer.gen.send(test)

            self.assertEqual(f(x), f(a))

        x = AlexArray(jnp.array(1))
        a1 = jnp.array(x)
        self.assertAllClose(1, a1)

        a2 = jnp.array(((x, x), [x, x]))
        self.assertAllClose(np.array(((1, 1), (1, 1))), a2)
