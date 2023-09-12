# Copyright 2019 The JAX Authors.
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
import functools
import pickle
import re

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import tree_util
from jax import flatten_util
from jax._src import test_util as jtu
from jax._src.tree_util import prefix_errors, flatten_one_level
import jax.numpy as jnp
from ..utils.timer_wrapper import jax_op_timer, partial_timed


def _dummy_func(*args, **kwargs):
    return


ATuple = collections.namedtuple("ATuple", ("foo", "bar"))


class ANamedTupleSubclass(ATuple):
    pass


ATuple2 = collections.namedtuple("ATuple2", ("foo", "bar"))
tree_util.register_pytree_node(
    ATuple2, lambda o: ((o.foo,), o.bar), lambda bar, foo: ATuple2(foo[0], bar)
)


class AnObject:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __repr__(self):
        return f"AnObject({self.x},{self.y},{self.z})"


tree_util.register_pytree_node(
    AnObject, lambda o: ((o.x, o.y), o.z), lambda z, xy: AnObject(xy[0], xy[1], z)
)


class AnObject2(AnObject):
    pass


tree_util.register_pytree_with_keys(
    AnObject2,
    lambda o: ((("x", o.x), ("y", o.y)), o.z),  # flatten_with_keys
    lambda z, xy: AnObject2(xy[0], xy[1], z),  # unflatten (no key involved)
)


@tree_util.register_pytree_node_class
class Special:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Special(x={self.x}, y={self.y})"

    def tree_flatten(self):
        return ((self.x, self.y), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __eq__(self, other):
        return type(self) is type(other) and (self.x, self.y) == (other.x, other.y)


@tree_util.register_pytree_with_keys_class
class SpecialWithKeys(Special):
    def tree_flatten_with_keys(self):
        return (
            ((tree_util.GetAttrKey("x"), self.x), (tree_util.GetAttrKey("y"), self.y)),
            None,
        )


@tree_util.register_pytree_node_class
class FlatCache:
    def __init__(self, structured, *, leaves=None, treedef=None):
        if treedef is None:
            leaves, treedef = tree_util.tree_flatten(structured)
        self._structured = structured
        self.treedef = treedef
        self.leaves = leaves

    def __hash__(self):
        return hash(self.structured)

    def __eq__(self, other):
        return self.structured == other.structured

    def __repr__(self):
        return f"FlatCache({self.structured!r})"

    @property
    def structured(self):
        if self._structured is None:
            self._structured = tree_util.tree_unflatten(self.treedef, self.leaves)
        return self._structured

    def tree_flatten(self):
        return self.leaves, self.treedef

    @classmethod
    def tree_unflatten(cls, meta, data):
        if not tree_util.all_leaves(data):
            data, meta = tree_util.tree_flatten(tree_util.tree_unflatten(meta, data))
        return FlatCache(None, leaves=data, treedef=meta)


TREES = (
    (None,),
    ((None,),),
    ((),),
    (([()]),),
    ((1, 2),),
    (((1, "foo"), ["bar", (3, None, 7)]),),
    ([3],),
    ([3, ATuple(foo=(3, ATuple(foo=3, bar=None)), bar={"baz": 34})],),
    ([AnObject(3, None, [4, "foo"])],),
    ([AnObject2(3, None, [4, "foo"])],),
    (Special(2, 3.0),),
    ({"a": 1, "b": 2},),
    (collections.OrderedDict([("foo", 34), ("baz", 101), ("something", -42)]),),
    (collections.defaultdict(dict, [("foo", 34), ("baz", 101), ("something", -42)]),),
    (ANamedTupleSubclass(foo="hello", bar=3.5),),
    (FlatCache(None),),
    (FlatCache(1),),
    (FlatCache({"a": [1, 2]}),),
)


TREE_STRINGS = (
    "PyTreeDef(None)",
    "PyTreeDef((None,))",
    "PyTreeDef(())",
    "PyTreeDef([()])",
    "PyTreeDef((*, *))",
    "PyTreeDef(((*, *), [*, (*, None, *)]))",
    "PyTreeDef([*])",
    (
        "PyTreeDef([*, CustomNode(namedtuple[ATuple], [(*, "
        "CustomNode(namedtuple[ATuple], [*, None])), {'baz': *}])])"
    ),
    "PyTreeDef([CustomNode(AnObject[[4, 'foo']], [*, None])])",
    "PyTreeDef([CustomNode(AnObject2[[4, 'foo']], [*, None])])",
    "PyTreeDef(CustomNode(Special[None], [*, *]))",
    "PyTreeDef({'a': *, 'b': *})",
)

# pytest expects "tree_util_test.ATuple"
STRS = []
for tree_str in TREE_STRINGS:
    tree_str = re.escape(tree_str)
    tree_str = tree_str.replace("__main__", ".*")
    STRS.append(tree_str)
TREE_STRINGS = STRS

LEAVES = (
    ("foo",),
    (0.1,),
    (1,),
    (object(),),
)

# All except those decorated by register_pytree_node_class
TREES_WITH_KEYPATH = (
    (None,),
    ((None,),),
    ((),),
    (([()]),),
    ((1, 0),),
    (((1, "foo"), ["bar", (3, None, 7)]),),
    ([3],),
    ([3, ATuple(foo=(3, ATuple(foo=3, bar=None)), bar={"baz": 34})],),
    ([AnObject2(3, None, [4, "foo"])],),
    (SpecialWithKeys(2, 3.0),),
    ({"a": 1, "b": 0},),
    (collections.OrderedDict([("foo", 34), ("baz", 101), ("something", -42)]),),
    (collections.defaultdict(dict, [("foo", 34), ("baz", 101), ("something", -42)]),),
    (ANamedTupleSubclass(foo="hello", bar=3.5),),
)


class TreeTest(jtu.JaxTestCase):
    def testPartialDoesNotMergeWithOtherPartials(self):
        def f(a, b, c):
            pass

        g = functools.partial(f, 2)
        timer = jax_op_timer()
        with timer:
            h = tree_util.Partial(g, 3)
            timer.gen.send(h)
        self.assertEqual(h.args, (3,))

    def testPartialFuncAttributeHasStableHash(self):
        # https://github.com/google/jax/issues/9429
        fun = functools.partial(print, 1)
        timer = jax_op_timer()
        with timer:
            p1 = tree_util.Partial(fun, 2)
            timer.gen.send(p1)
        timer = jax_op_timer()
        with timer:
            p2 = tree_util.Partial(fun, 2)
            timer.gen.send(p2)
        self.assertEqual(fun, p1.func)
        self.assertEqual(p1.func, fun)
        self.assertEqual(p1.func, p2.func)
        self.assertEqual(hash(p1.func), hash(p2.func))
