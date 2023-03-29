"""Tests for utilities working with arbitrarily nested structures."""

import collections
import collections.abc
import time
from typing import NamedTuple

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.util import nest

try:
  import attr  # pylint:disable=g-import-not-at-top
except ImportError:
  attr = None


class _CustomMapping(collections.abc.Mapping):

  def __init__(self, *args, **kwargs):
    self._wrapped = dict(*args, **kwargs)

  def __getitem__(self, key):
    return self._wrapped[key]

  def __iter__(self):
    return iter(self._wrapped)

  def __len__(self):
    return len(self._wrapped)


class _CustomList(list):
  pass


class _CustomSequenceThatRaisesException(collections.abc.Sequence):

  def __len__(self):
    return 1

  def __getitem__(self, item):
    raise ValueError("Cannot get item: %s" % item)


class NestTest(parameterized.TestCase, test.TestCase):

  PointXY = collections.namedtuple("Point", ["x", "y"])  # pylint: disable=invalid-name
  unsafe_map_pattern = ("nest cannot guarantee that it is safe to map one to "
                        "the other.")
  bad_pack_pattern = ("Attempted to pack value:\n  .+\ninto a structure, but "
                      "found incompatible type `<(type|class) 'str'>` instead.")

  if attr:
    class BadAttr(object):
      """Class that has a non-iterable __attrs_attrs__."""
      __attrs_attrs__ = None

    @attr.s
    class SampleAttr(object):
      field1 = attr.ib()
      field2 = attr.ib()

    @attr.s
    class UnsortedSampleAttr(object):
      field3 = attr.ib()
      field1 = attr.ib()
      field2 = attr.ib()

  def testMapStructureUpTo(self):
    # Named tuples.
    ab_tuple = collections.namedtuple("ab_tuple", "a, b")
    op_tuple = collections.namedtuple("op_tuple", "add, mul")
    inp_val = ab_tuple(a=2, b=3)
    inp_ops = ab_tuple(a=op_tuple(add=1, mul=2), b=op_tuple(add=2, mul=3))
    out = nest.map_structure_up_to(
        inp_val, lambda val, ops: (val + ops.add) * ops.mul, inp_val, inp_ops)
    self.assertEqual(out.a, 6)
    self.assertEqual(out.b, 15)

    # Lists.
    data_list = [[2, 4, 6, 8], [[1, 3, 5, 7, 9], [3, 5, 7]]]
    name_list = ["evens", ["odds", "primes"]]
    out = nest.map_structure_up_to(
        name_list, lambda name, sec: "first_{}_{}".format(len(sec), name),
        name_list, data_list)
    self.assertEqual(out, ["first_4_evens", ["first_5_odds", "first_3_primes"]])

    # Dicts.
    inp_val = dict(a=2, b=3)
    inp_ops = dict(a=dict(add=1, mul=2), b=dict(add=2, mul=3))
    out = nest.map_structure_up_to(
        inp_val,
        lambda val, ops: (val + ops["add"]) * ops["mul"], inp_val, inp_ops)
    self.assertEqual(out["a"], 6)
    self.assertEqual(out["b"], 15)

    # Non-equal dicts.
    inp_val = dict(a=2, b=3)
    inp_ops = dict(a=dict(add=1, mul=2), c=dict(add=2, mul=3))
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        nest._SHALLOW_TREE_HAS_INVALID_KEYS.format(["b"])):
      nest.map_structure_up_to(
          inp_val,
          lambda val, ops: (val + ops["add"]) * ops["mul"], inp_val, inp_ops)

    # Dict+custom mapping.
    inp_val = dict(a=2, b=3)
    inp_ops = _CustomMapping(a=dict(add=1, mul=2), b=dict(add=2, mul=3))
    out = nest.map_structure_up_to(
        inp_val,
        lambda val, ops: (val + ops["add"]) * ops["mul"], inp_val, inp_ops)
    self.assertEqual(out["a"], 6)
    self.assertEqual(out["b"], 15)

    # Non-equal dict/mapping.
    inp_val = dict(a=2, b=3)
    inp_ops = _CustomMapping(a=dict(add=1, mul=2), c=dict(add=2, mul=3))
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        nest._SHALLOW_TREE_HAS_INVALID_KEYS.format(["b"])):
      nest.map_structure_up_to(
          inp_val,
          lambda val, ops: (val + ops["add"]) * ops["mul"], inp_val, inp_ops)

  
if __name__ == "__main__":
  test.main()
