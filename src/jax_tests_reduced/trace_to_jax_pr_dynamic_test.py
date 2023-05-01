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

import unittest
from collections import namedtuple
from functools import partial
import gc
import itertools as it
import operator

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import lax
from jax import numpy as jnp
from jax import jvp, linearize, vjp, jit, make_jaxpr
from jax.api_util import flatten_fun_nokwargs
from jax.config import config
from jax.tree_util import (
    tree_flatten,
    tree_unflatten,
    tree_map,
    tree_reduce,
    tree_leaves,
)

from jax._src import core
from jax._src import linear_util as lu
from jax._src import util
from jax._src import test_util as jtu
from jax._src.core import UnshapedArray, ShapedArray, DBIdx
from jax._src.interpreters import partial_eval as pe
from jax._src.lax import lax as lax_internal
from jax._src.lax import control_flow as lax_control_flow

config.parse_flags_with_absl()

_ = pe.PartialVal.unknown(UnshapedArray(np.float32))
__ = pe.PartialVal.unknown(ShapedArray((), np.float32))


def call(f, *args):
    return jit(f)(*args)


@util.curry
def core_call(f, *args):
    args, in_tree = tree_flatten(args)
    f, out_tree = flatten_fun_nokwargs(lu.wrap_init(f), in_tree)
    out = core.call_p.bind(f, *args)
    return tree_unflatten(out_tree(), out)


@util.curry
def core_closed_call(f, *args):
    args, in_tree = tree_flatten(args)
    f, out_tree = flatten_fun_nokwargs(lu.wrap_init(f), in_tree)
    out = core.closed_call_p.bind(f, *args)
    return tree_unflatten(out_tree(), out)


def simple_fun(x, y):
    return jnp.sin(x * y)


def simple_fun_fanout(x, y):
    return jnp.sin(x * y) * x


def fun_with_call(x):
    return call(jnp.sin, x)


def fun_with_nested_calls(x):
    def f(y):
        y2 = jnp.sin(y) + 1.0 + (2.0 * x)

        @jit
        def g(z):
            return y2 * z * x + (x * y)

        return call(g, y)

    return call(f, x)


def error(*args):
    def f(*args):
        assert False

    return f


def fun_with_nested_calls_2(x):
    def bar(y):
        def baz(w):
            q = call(lambda x: y, x)
            q = q + call(lambda: y)
            q = q + call(lambda y: w + y, y)
            q = call(lambda w: call(jnp.sin, x) * y, 1.0) + q
            return q

        p, t = jvp(baz, (x + 1.0,), (y,))
        return t + (x * p)

    return call(bar, x)


def fun_call_jitted(x):
    @jit
    def g(z):
        return x * z

    return call(g, x)


def fun_with_two_calls(x):
    return call(jnp.sin, x) + call(jnp.cos, x)


def fun_with_call_closure(x):
    def foo(y, z):
        return (x * x) * jnp.sin(y) * z

    return call(foo, x, jnp.cos(x)) + x


def product_io_fun(x, y):
    xa = x["a"]
    xb = x["b"]
    y1, (y2, y3) = y
    return jnp.sin(xa + y2), [xb, (y1, y3)]


_rng = np.random.RandomState(42)
R = _rng.randn
CallSpec = namedtuple("CallSpec", ["fun", "args"])
test_specs_base = [
    CallSpec(simple_fun, (R(3, 2), R(3, 2))),
    CallSpec(simple_fun_fanout, (R(3, 2), R(3, 2))),
    CallSpec(
        product_io_fun, ({"a": R(2, 2), "b": R(2, 2)}, (R(2, 2), (R(2, 2), R(2, 2))))
    ),
    CallSpec(fun_with_call, (R(3, 2),)),
    CallSpec(fun_with_two_calls, (R(3, 2),)),
    CallSpec(fun_with_call_closure, (R(3, 2),)),
    CallSpec(
        fun_call_jitted,
        (
            R(
                1,
            ),
        ),
    ),
    CallSpec(fun_with_nested_calls, (R(),)),
    CallSpec(fun_with_nested_calls, (R(3, 2),)),
    CallSpec(fun_with_nested_calls_2, (R(1, 2),)),
]


def jvp_unlinearized(f, primals, tangents):
    out, jvp = linearize(f, *primals)
    return out, jvp(*tangents)


test_specs = []
for ts in test_specs_base:
    test_specs.append(ts)
    test_specs.append(CallSpec(partial(jvp, ts.fun), (ts.args, ts.args)))
    test_specs.append(CallSpec(jit(ts.fun), ts.args))
    test_specs.append(CallSpec(jit(jit(ts.fun)), ts.args))
    test_specs.append(CallSpec(core_call(ts.fun), ts.args))
    test_specs.append(CallSpec(core_call(jit(ts.fun)), ts.args))
    test_specs.append(CallSpec(core_call(core_call(ts.fun)), ts.args))
    test_specs.append(CallSpec(core_closed_call(ts.fun), ts.args))
    test_specs.append(CallSpec(core_closed_call(jit(ts.fun)), ts.args))
    test_specs.append(CallSpec(core_closed_call(core_closed_call(ts.fun)), ts.args))
    test_specs.append(CallSpec(partial(jvp_unlinearized, ts.fun), (ts.args, ts.args)))


def fwd_deriv(f):
    def df(x):
        return jvp(f, (x,), (1.0,))[1]

    return df


@jtu.with_config(jax_dynamic_shapes=True)
class DynamicShapesTest(jtu.JaxTestCase):
    def test_staging_basic(self):
        n = core.ShapedArray((), jnp.dtype("int32"), weak_type=False)
        a = core.DShapedArray((DBIdx(0),), jnp.dtype("float32"), weak_type=False)
        b = core.DShapedArray((DBIdx(0),), jnp.dtype("float32"), weak_type=False)

        @lu.wrap_init
        def f(x, y):
            return x, y

        jaxpr, _, _ = pe.trace_to_jaxpr_dynamic(
            f, [n, a, b], keep_inputs=[False, True, True]
        )

        self.assertLen(jaxpr.invars, 3)
        self.assertEqual((jaxpr.invars[0],), jaxpr.invars[1].aval.shape)
        self.assertEqual((jaxpr.invars[0],), jaxpr.invars[2].aval.shape)

        self.assertLen(jaxpr.outvars, 2)
        self.assertEqual((jaxpr.invars[0],), jaxpr.outvars[0].aval.shape)
        self.assertEqual((jaxpr.invars[0],), jaxpr.outvars[1].aval.shape)
