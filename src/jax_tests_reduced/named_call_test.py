# Copyright 2020 The JAX Authors.
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

from functools import partial
import itertools
import logging
import os
import re
import threading
import time
from typing import Callable, Optional, Sequence
import unittest
from unittest import skip, SkipTest

from absl.testing import absltest

import jax
from jax import ad_checkpoint
from jax._src import core
from jax.config import config
from jax import dtypes
from jax.experimental import host_callback as hcb
from jax.sharding import PartitionSpec as P
from jax.experimental import pjit
from jax import lax
from jax import numpy as jnp
from jax._src import test_util as jtu
from jax import tree_util
from jax._src import xla_bridge
from jax._src.lib import xla_client
from ..utils.timer_wrapper import jax_op_timer

xops = xla_client.ops

import numpy as np

config.parse_flags_with_absl()
FLAGS = config.FLAGS


class _TestingOutputStream:
    """Use as `output_stream` for tests."""

    def __init__(self):
        self._output = []
        self._test_method_name = None

    def write(self, what: str) -> None:
        logging.info(f"output_stream[{self._test_method_name}]: {what}")
        self._output.append(what)

    @property
    def output(self):
        return "".join(self._output)

    @property
    def output_sorted_by_device(self):
        # Assume that the output is a sequence of strings including metadata
        # and data, with metadata containing `device: xxx`
        by_device = []  # each element is a pair (device, str_list)
        for s in self._output:
            m = re.match(r".*device: (\S+)", s)
            if m:
                by_device.append((m.group(1), []))
            assert by_device, f"output does not include 'device:': {self._output}"
            by_device[-1][1].append(s)

        sorted_by_device = sorted(by_device, key=lambda x: x[0])
        return "\n".join(itertools.chain(*[s[1] for s in sorted_by_device]))

    def __str__(self):
        return "TestingOutputStream"

    def reset(self):
        self._output = []


testing_stream = _TestingOutputStream()


def fun1(a):
    """Function used for several `id_tap` tests."""
    y = hcb.id_print(a * 2.0, what="a * 2", output_stream=testing_stream)
    y = hcb.id_print(y * 3.0, what="y * 3", output_stream=testing_stream, result=y)
    return y**2  # Some computation to make the gradient interesting


def fun1_equiv(a):  # Numerical equivalent of fun1
    return (a * 2.0) ** 2


def maybe_print(
    do_print: bool,
    arg,
    what: str,
    tap_with_device: Optional[bool] = False,
    device_index: int = 0,
):
    """Conditionally print on testing_string"""
    if do_print:
        return hcb.id_print(
            arg,
            what=what,
            output_stream=testing_stream,
            tap_with_device=tap_with_device,
            device_index=device_index,
        )
    else:
        return arg


def local_devices():
    # Tests require using not more than 2 devices.
    return jax.local_devices()[:2]


ignore_jit_of_pmap_warning = partial(jtu.ignore_warning, message=".*jit-of-pmap.*")


def assertMultiLineStrippedEqual(tst: jtu.JaxTestCase, expected: str, what: str):
    """A variant that preprocesses the string to eliminate non-determinism in
    floating point values, and several uninteresting id_tap primitive params.
    """

    # Sometimes we get floating points in the output; we round them
    def repl_floats(match_group):
        matched = match_group.group(0)
        if matched == ".":
            return matched
        x = np.around(float(matched), decimals=2)
        return f"{x:.2f}"

    what = re.sub(r"\-?\d+\.[\-\def]*", repl_floats, what)
    what = re.sub(r"output_stream=[^\]\n,]*,?", "", what)
    what = re.sub(r"threshold=[^\]\n,]*,?", "", what)
    what = re.sub(r"bwd=[^\]\n]*", "", what)
    what = re.sub(r"out_trees=[^\]\n]*", "", what)
    what = re.sub(r"fwd_jaxpr_thunk=[^\]\n]*", "", what)
    what = re.sub(r"jvp_jaxpr_thunk=[^\]\n]*", "", what)
    # Empty lines
    what = re.sub(r"^\s*\n", "", what, flags=re.MULTILINE)

    def repl_func(match_group):
        matched = match_group.group(3)
        if "function _print_consumer" in matched:
            return match_group.group(1) + "=_print"
        else:
            return match_group.group(1) + "=..."

    what = re.sub(r"((tap_func_)|(callback))=([^\]\n,]*),?", repl_func, what)
    tst.assertMultiLineStrippedEqual(expected, what)


def helper_set_hlo_dump():
    flags_str = os.getenv("XLA_FLAGS", "")
    import shutil

    dump_dir = "/tmp/xla_dump"
    os.environ["XLA_FLAGS"] = f"{flags_str} --xla_dump_to={dump_dir}"
    if os.path.isdir(dump_dir):
        logging.warning("Deleting old XLA dump directory %s", dump_dir)
        shutil.rmtree(dump_dir)
    logging.warning("Setting XLA dump directory %s", dump_dir)
    # Clear any cached backends so new CPU backend will pick up the env var.
    xla_bridge.get_backend.cache_clear()


def helper_print_optimized_hlo(fun, *args):
    backend = xla_bridge.get_backend(platform=jtu.device_under_test())
    c = jax.jit(fun, backend=backend.platform).lower(*args)
    logging.info(re.sub(r", metadata.*", "", c.compile().as_text()))


def helper_log_ir(name, f_jax, *args, num_partitions=None, strip_metadata=False):
    logging.info(f"Jaxpr[{name}]: {jax.make_jaxpr(f_jax)(*args)}")
    jax_comp = f_jax.lower(*args)
    logging.info(f"HLO[{name}]: {jax_comp.compiler_ir(dialect='hlo').as_hlo_text()}")
    jax_optimized_hlo = jax_comp.compile().as_text()
    if strip_metadata:
        jax_optimized_hlo = re.sub(r", metadata.*", "", jax_optimized_hlo)
    logging.info(f"Optimized HLO[{name}]: {jax_optimized_hlo}")


prev_xla_flags = None


def setUpModule():
    global prev_xla_flags
    # This will control the CPU devices. On TPU we always have 2 devices
    prev_xla_flags = jtu.set_host_platform_device_count(2)


# Reset to previous configuration in case other test modules will be run.
def tearDownModule():
    prev_xla_flags()


def assertMultiDeviceOutputEqual(tst: jtu.JaxTestCase, expected_2CPUs: str):
    """Check that the multi-device output is equal to the expected.

    The tests run with 2 devices if available, otherwise 1 device.
    We adjust the expected output here for 1 device.

    Args:
      expected_2CPUs: the expected output for 2 CPUs. If there is only
        one device, this is trimmed to the first device. If the current
        device_under_test is not a CPU, then we change the names
    """
    expected = expected_2CPUs
    if len(local_devices()) == 1:
        start_device_1 = expected.find("device: cpu:1")
        if start_device_1 >= 0:
            expected = expected[0:start_device_1]

    def replace_device_name(m) -> str:
        return str(local_devices()[int(m.group(1))])

    expected = re.sub(r"cpu:(\d+)", replace_device_name, expected)
    what = testing_stream.output_sorted_by_device
    return assertMultiLineStrippedEqual(tst, expected, what)


class HostCallbackTapTest(jtu.JaxTestCase):
    def setUp(self):
        if jtu.device_under_test() == "gpu" and jax.device_count() > 1:
            raise SkipTest("host_callback broken on multi-GPU platforms (#6447)")
        if xla_bridge.using_pjrt_c_api():
            raise SkipTest("host_callback not implemented in PJRT C API")
        super().setUp()

    def assertRewrite(
        self,
        expected: str,
        func: Callable,
        args: Sequence,
        has_input_token=True,
        has_output_token=True,
    ):
        """Check that the rewrite of func(*args) matches expected."""
        jaxpr = jax.make_jaxpr(func)(*args)
        rewritten = hcb._rewrite_closed_jaxpr(
            jaxpr, has_input_token, has_output_token  # noqa: F841
        )
        # Since it is somewhat annoying to update the Jaxpr assertions when we change
        # the Jaxpr printing, we do not check these by default. It is recommended that
        # before making changes to the code generation and Jaxpr rewriting, turn on
        # the checking, update the expected Jaxpr, and then make the changes.
        # assertMultiLineStrippedEqual(self, expected, str(rewritten))
        del rewritten

    def test_tap_named_call(self):
        def tap_scalar(init, do_print=False):
            @partial(jax.named_call, name="step")
            def step(acc, step_nr):
                acc = acc + step_nr
                maybe_print(do_print, step_nr, what="step_nr")
                return acc, None

            timer = jax_op_timer()
            with timer:
                function = jax.named_call(step, name="step")
                timer.gen.send(function)
            return lax.scan(function, init, np.arange(2))

        self.assertRewrite(
            """
        { lambda a ; b d e.
            let c = scan[ jaxpr={ lambda  ; a b.
                                let c = named_call[ call_jaxpr={ lambda  ; a b.
                                                                    let c = add a b
                                                                    in (c,) }
                                                    name=step ] a b
                                in (c,) }
                        length=2
                        linear=(False, False)
                        num_carry=1
                        num_consts=0
                        reverse=False
                        unroll=1 ] b a
            in (c, d, e) }""",
            tap_scalar,
            [np.int32(3)],
        )
