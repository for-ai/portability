
import gc
import os
import threading
import weakref

from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import full_type_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as eager_function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import versions
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import resources
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
import tensorflow.python.ops.gradients  # pylint: disable=unused-import
from tensorflow.python.platform import googletest
from tensorflow.python.util import compat
from ..utils.tensorflow_contexts import PortabilityTestCase
import tensorflow as tf
class ObjectWithName(object):
    
  def __init__(self, name):
    self._name = name

  @property
  def name(self):
    return self._name

class CollectionTest(PortabilityTestCase, test_util.TensorFlowTestCase):

  def test_get_collections(self):
    with self.cached_session(): 
      a = tf.constant([1, 2, 3])
      tf.print("*** test",a.device) 
    g = ops.Graph()
    self.assertSequenceEqual(g.collections, [])
    g.add_to_collection("key", 12)
    g.add_to_collection("key", 15)
    self.assertSequenceEqual(g.collections, ["key"])
    g.add_to_collection("other", "foo")
    self.assertSequenceEqual(sorted(g.collections), ["key", "other"])
    self.assertSequenceEqual(
        sorted(g.get_all_collection_keys()), ["key", "other"])

  def test_add_to_collection(self):
    g = ops.Graph()
    g.add_to_collection("key", 12)
    g.add_to_collection("other", "foo")
    g.add_to_collection("key", 34)

    # Note that only blank1 is returned.
    g.add_to_collection("blah", 27)
    blank1 = ObjectWithName("prefix/foo")
    g.add_to_collection("blah", blank1)
    blank2 = ObjectWithName("junk/foo")
    g.add_to_collection("blah", blank2)

    self.assertEqual([12, 34], g.get_collection("key"))
    self.assertEqual([], g.get_collection("nothing"))
    self.assertEqual([27, blank1, blank2], g.get_collection("blah"))
    self.assertEqual([blank1], g.get_collection("blah", "prefix"))
    self.assertEqual([blank1], g.get_collection("blah", ".*x"))

    # Make sure that get_collection() returns a first-level
    # copy of the collection, while get_collection_ref() returns
    # the original list.
    other_collection_snapshot = g.get_collection("other")
    other_collection_ref = g.get_collection_ref("other")
    self.assertEqual(["foo"], other_collection_snapshot)
    self.assertEqual(["foo"], other_collection_ref)
    g.add_to_collection("other", "bar")
    self.assertEqual(["foo"], other_collection_snapshot)
    self.assertEqual(["foo", "bar"], other_collection_ref)
    self.assertEqual(["foo", "bar"], g.get_collection("other"))
    self.assertTrue(other_collection_ref is g.get_collection_ref("other"))

    # Verify that getting an empty collection ref returns a modifiable list.
    empty_coll_ref = g.get_collection_ref("empty")
    self.assertEqual([], empty_coll_ref)
    empty_coll = g.get_collection("empty")
    self.assertEqual([], empty_coll)
    self.assertFalse(empty_coll is empty_coll_ref)
    empty_coll_ref2 = g.get_collection_ref("empty")
    self.assertTrue(empty_coll_ref2 is empty_coll_ref)
    # Add to the collection.
    empty_coll_ref.append("something")
    self.assertEqual(["something"], empty_coll_ref)
    self.assertEqual(["something"], empty_coll_ref2)
    self.assertEqual([], empty_coll)
    self.assertEqual(["something"], g.get_collection("empty"))
    empty_coll_ref3 = g.get_collection_ref("empty")
    self.assertTrue(empty_coll_ref3 is empty_coll_ref)

  def test_add_to_collections_uniquify(self):
    g = ops.Graph()
    g.add_to_collections([1, 2, 1], "key")
    # Make sure "key" is not added twice
    self.assertEqual(["key"], g.get_collection(1))

  def test_add_to_collections_from_list(self):
    g = ops.Graph()
    g.add_to_collections(["abc", "123"], "key")
    self.assertEqual(["key"], g.get_collection("abc"))
    self.assertEqual(["key"], g.get_collection("123"))

  def test_add_to_collections_from_tuple(self):
    g = ops.Graph()
    g.add_to_collections(("abc", "123"), "key")
    self.assertEqual(["key"], g.get_collection("abc"))
    self.assertEqual(["key"], g.get_collection("123"))

  def test_add_to_collections_from_generator(self):
    g = ops.Graph()

    def generator():
      yield "abc"
      yield "123"

    g.add_to_collections(generator(), "key")
    self.assertEqual(["key"], g.get_collection("abc"))
    self.assertEqual(["key"], g.get_collection("123"))

  def test_add_to_collections_from_set(self):
    g = ops.Graph()
    g.add_to_collections(set(["abc", "123"]), "key")
    self.assertEqual(["key"], g.get_collection("abc"))
    self.assertEqual(["key"], g.get_collection("123"))

  def test_add_to_collections_from_string(self):
    g = ops.Graph()
    g.add_to_collections("abc", "key")
    self.assertEqual(["key"], g.get_collection("abc"))

  def test_default_graph(self):
    with ops.Graph().as_default():
      ops.add_to_collection("key", 90)
      ops.add_to_collection("key", 100)
      # Collections are ordered.
      self.assertEqual([90, 100], ops.get_collection("key"))


ops.NotDifferentiable("FloatOutput")

if __name__ == "__main__":
  googletest.main()