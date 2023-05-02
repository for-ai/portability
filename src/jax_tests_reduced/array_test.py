# Copyright 2021 The JAX Authors.
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
"""Tests for GlobalDeviceArray."""

import contextlib
import math
import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import array, core, dispatch, prng
from jax._src import test_util as jtu
from jax._src import xla_bridge as xb
from jax._src.lib import xla_client as xc
from jax._src.util import safe_zip
from jax.config import config
from jax.experimental import multihost_utils
from jax.experimental.pjit import pjit
from jax.experimental.serialize_executable import (deserialize_and_load,
                                                   serialize)
from jax.interpreters import pxla
from jax.sharding import PartitionSpec as P

from ..utils.timer_wrapper import jax_op_timer, partial_timed

config.parse_flags_with_absl()


prev_xla_flags = None

with contextlib.suppress(ImportError):
    import pytest

    pytestmark = pytest.mark.multiaccelerator


# Run all tests with 8 CPU devices.
def setUpModule():
    global prev_xla_flags
    prev_xla_flags = os.getenv("XLA_FLAGS")
    flags_str = prev_xla_flags or ""
    # Don't override user-specified device count, or other XLA flags.
    if "xla_force_host_platform_device_count" not in flags_str:
        os.environ["XLA_FLAGS"] = (
            flags_str + " --xla_force_host_platform_device_count=8"
        )
    # Clear any cached backends so new CPU backend will pick up the env var.
    xb.get_backend.cache_clear()


# Reset to previous configuration in case other test modules will be run.
def tearDownModule():
    if prev_xla_flags is None:
        del os.environ["XLA_FLAGS"]
    else:
        os.environ["XLA_FLAGS"] = prev_xla_flags
    xb.get_backend.cache_clear()


def create_array(shape, sharding, global_data=None):
    if global_data is None:
        global_data = np.arange(math.prod(shape)).reshape(shape)

    return (
        array.make_array_from_callback(shape, sharding, lambda idx: global_data[idx]),
        global_data,
    )


class JaxArrayTest(jtu.JaxTestCase):
    def test_jnp_array(self):
        timer = jax_op_timer()
        with timer:
            arr = jnp.array([1, 2, 3])
            timer.gen.send(arr)

        self.assertIsInstance(arr, array.ArrayImpl)
        self.assertTrue(dispatch.is_single_device_sharding(arr.sharding))
        self.assertEqual(arr._committed, False)
        self.assertFalse(arr.weak_type)
