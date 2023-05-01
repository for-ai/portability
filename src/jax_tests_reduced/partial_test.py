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


class NamedCallTest(jtu.JaxTestCase):
    def test_partial_eval(self):
        f = api.named_call(lambda x, y: y if x else None, name="test")
        f = jax.jit(functools.partial(f, True))
        out = f(5)
        self.assertEqual(out, 5)
