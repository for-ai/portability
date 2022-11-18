# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for lookup ops."""
import os
import tempfile
import unittest

from absl.testing import parameterized
import numpy as np

from tensorflow.python import tf2
from tensorflow.python.checkpoint import checkpoint as trackable
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.checkpoint import util as checkpoint_util
from tensorflow.python.client import session
from tensorflow.python.data.experimental.ops import counter
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load as saved_model_load
from tensorflow.python.saved_model import save as saved_model_save
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import autotrackable
from tensorflow.python.training import saver
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat


class BaseLookupTableTest(test.TestCase):

  def getHashTable(self):
    if tf2.enabled():
      return lookup_ops.StaticHashTable
    else:
      return lookup_ops.StaticHashTableV1

  def getVocabularyTable(self):
    if tf2.enabled():
      return lookup_ops.StaticVocabularyTable
    else:
      return lookup_ops.StaticVocabularyTableV1

  def initialize_table(self, table):
    if not tf2.enabled():
      self.evaluate(table.initializer)


SKIP_ANONYMOUS_IN_TF1_REASON = (
    "In v1 graph mode, each self.evaluate call will execute the handle "
    "creation op (e.g. AnonymousHashTable) which will create a new table "
    "resource unrelated to other self.evaluate calls, so we can't test "
    "anonymous resources with self.evaluate ."
)


@parameterized.named_parameters(
    (f"_{is_anonymous}", is_anonymous) for is_anonymous in [False, True])
class StaticHashTableTest(BaseLookupTableTest, parameterized.TestCase):

  def testStaticHashTable(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.assertEqual(table._is_anonymous, is_anonymous)
    self.initialize_table(table)

    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant(["brain", "salad", "tank"])
    output = table.lookup(input_string)
    self.assertAllEqual([3], output.get_shape())

    result = self.evaluate(output)
    self.assertAllEqual([0, 1, -1], result)

    exported_keys_tensor, exported_values_tensor = table.export()

    self.assertItemsEqual([b"brain", b"salad", b"surgery"],
                          self.evaluate(exported_keys_tensor))
    self.assertItemsEqual([0, 1, 2], self.evaluate(exported_values_tensor))

  def testStaticHashTableFindHighRank(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant([["brain", "salad"],
                                         ["tank", "tarkus"]])
    output = table.lookup(input_string)

    result = self.evaluate(output)
    self.assertAllEqual([[0, 1], [-1, -1]], result)

  def testStaticHashTableInitWithPythonArrays(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = ["brain", "salad", "surgery"]
    values = [0, 1, 2]
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(
            keys, values, value_dtype=dtypes.int64),
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant(["brain", "salad", "tank"])
    output = table.lookup(input_string)

    result = self.evaluate(output)
    self.assertAllEqual([0, 1, -1], result)

  def testStaticHashTableInitWithNumPyArrays(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = np.array(["brain", "salad", "surgery"], dtype=np.str_)
    values = np.array([0, 1, 2], dtype=np.int64)
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant(["brain", "salad", "tank"])
    output = table.lookup(input_string)

    result = self.evaluate(output)
    self.assertAllEqual([0, 1, -1], result)

  def testMultipleStaticHashTables(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)

    table1 = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)
    table2 = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)
    table3 = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)

    self.initialize_table(table1)
    self.initialize_table(table2)
    self.initialize_table(table3)
    self.assertAllEqual(3, self.evaluate(table1.size()))
    self.assertAllEqual(3, self.evaluate(table2.size()))
    self.assertAllEqual(3, self.evaluate(table3.size()))

    input_string = constant_op.constant(["brain", "salad", "tank"])
    output1 = table1.lookup(input_string)
    output2 = table2.lookup(input_string)
    output3 = table3.lookup(input_string)

    out1, out2, out3 = self.evaluate([output1, output2, output3])
    self.assertAllEqual([0, 1, -1], out1)
    self.assertAllEqual([0, 1, -1], out2)
    self.assertAllEqual([0, 1, -1], out3)

  def testStaticHashTableWithTensorDefault(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = constant_op.constant(-1, dtypes.int64)
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    input_string = constant_op.constant(["brain", "salad", "tank"])
    output = table.lookup(input_string)

    result = self.evaluate(output)
    self.assertAllEqual([0, 1, -1], result)

  def testStaticHashTableGetItem(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = constant_op.constant(-1, dtypes.int64)
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    input_string = constant_op.constant(["brain", "salad", "tank"])
    output = table[input_string]

    result = self.evaluate(output)
    self.assertAllEqual([0, 1, -1], result)

  def testStaticHashTableWithSparseTensorInput(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = constant_op.constant(-1, dtypes.int64)
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    sp_indices = [[0, 0], [0, 1], [1, 0]]
    sp_shape = [2, 2]
    input_tensor = sparse_tensor.SparseTensor(
        constant_op.constant(sp_indices, dtypes.int64),
        constant_op.constant(["brain", "salad", "tank"]),
        constant_op.constant(sp_shape, dtypes.int64))
    output = table.lookup(input_tensor)

    out_indices, out_values, out_shape = self.evaluate(output)

    self.assertAllEqual([0, 1, -1], out_values)
    self.assertAllEqual(sp_indices, out_indices)
    self.assertAllEqual(sp_shape, out_shape)

  def testStaticHashTableWithRaggedTensorInput(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = constant_op.constant(-1, dtypes.int64)
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    row_splits = [0, 2, 3]
    input_tensor = ragged_tensor.RaggedTensor.from_row_splits(
        constant_op.constant(["brain", "salad", "tank"]),
        constant_op.constant(row_splits, dtypes.int64))
    output = table.lookup(input_tensor)

    out = self.evaluate(output)

    self.assertAllEqual([0, 1, -1], out.values)
    self.assertAllEqual(row_splits, out.row_splits)

  def testSignatureMismatch(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    # Ref types do not produce a lookup signature mismatch.
    input_string_ref = variables.Variable("brain")
    self.evaluate(input_string_ref.initializer)
    self.assertEqual(0, self.evaluate(table.lookup(input_string_ref)))

    input_string = constant_op.constant([1, 2, 3], dtypes.int64)
    with self.assertRaises(TypeError):
      table.lookup(input_string)

    with self.assertRaises(TypeError):
      self.getHashTable()(
          lookup_ops.KeyValueTensorInitializer(keys, values),
          "UNK",
          experimental_is_anonymous=is_anonymous)

  def testDTypes(self, is_anonymous):
    default_val = -1
    with self.assertRaises(TypeError):
      self.getHashTable()(
          lookup_ops.KeyValueTensorInitializer(["a"], [1], [dtypes.string],
                                               dtypes.int64),
          default_val,
          experimental_is_anonymous=is_anonymous)

  @test_util.run_v1_only("(Cached) Sessions not available in TF2.0")
  def testNotInitialized(self, is_anonymous):
    with self.cached_session():
      default_val = -1
      table = self.getHashTable()(
          lookup_ops.KeyValueTensorInitializer(["a"], [1],
                                               value_dtype=dtypes.int64),
          default_val,
          experimental_is_anonymous=is_anonymous)

      input_string = constant_op.constant(["brain", "salad", "surgery"])
      output = table.lookup(input_string)

      with self.assertRaisesOpError("Table not initialized"):
        self.evaluate(output)

  @test_util.run_v1_only("(Cached) Sessions not available in TF2.0")
  def testInitializeTwice(self, is_anonymous):
    with self.cached_session():
      default_val = -1
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = self.getHashTable()(
          lookup_ops.KeyValueTensorInitializer(keys, values),
          default_val,
          experimental_is_anonymous=is_anonymous)
      self.initialize_table(table)
      # Make sure that initializing twice doesn't throw any errors.
      self.initialize_table(table)

  def testInitializationWithInvalidDimensions(self, is_anonymous):
    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2, 3, 4], dtypes.int64)

    raised_error = ValueError
    if context.executing_eagerly():
      raised_error = errors_impl.InvalidArgumentError
    with self.assertRaises(raised_error):
      self.getHashTable()(
          lookup_ops.KeyValueTensorInitializer(keys, values),
          default_val,
          experimental_is_anonymous=is_anonymous)

  @test_util.run_v1_only("Sessions not available in TF2.0")
  def testMultipleSessions(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    # Start a server
    server = server_lib.Server({"local0": ["localhost:0"]},
                               protocol="grpc",
                               start=True)
    # Create two sessions sharing the same state
    session1 = session.Session(server.target)
    session2 = session.Session(server.target)

    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        name="t1",
        experimental_is_anonymous=is_anonymous)

    # Init the table in the first session.
    with session1:
      self.initialize_table(table)
      self.assertAllEqual(3, self.evaluate(table.size()))

    # Init the table in the second session and verify that we do not get a
    # "Table already initialized" error.
    with session2:
      self.evaluate(table.initializer)
      self.assertAllEqual(3, self.evaluate(table.size()))

  @test_util.run_v2_only
  def testImportedHashTable(self, is_anonymous):
    g = ops.Graph()
    with g.as_default():
      t = lookup_ops.StaticHashTable(
          lookup_ops.KeyValueTensorInitializer(["a"], [1]),
          2)
      init_op = t._init_op
      op = t.lookup(ops.convert_to_tensor(["a"]))
      meta_graph = saver.export_meta_graph()

    def f():
      saver.import_meta_graph(meta_graph)
      return ops.get_default_graph().get_tensor_by_name(op.name)

    wrapped = wrap_function.wrap_function(f, [])
    pruned_init_fn = wrapped.prune(
        (), [wrapped.graph.get_operation_by_name(init_op.name)])
    self.evaluate(pruned_init_fn())
    self.assertAllEqual([1], wrapped())

  def testStaticHashTableInt32String(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = "n/a"
    keys = constant_op.constant([0, 1, 2], dtypes.int32)
    values = constant_op.constant(["brain", "salad", "surgery"])
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    input_tensor = constant_op.constant([0, 1, -1])
    output = table.lookup(input_tensor)

    result = self.evaluate(output)
    self.assertAllEqual([b"brain", b"salad", b"n/a"], result)

  def testTableUseInFunction(self, is_anonymous):
    if not context.executing_eagerly():
      self.skipTest("Only Eager mode test.")
    keys = constant_op.constant([0, 1, 2], dtypes.int32)
    values = constant_op.constant(["brain", "salad", "surgery"])
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        "n/a",
        experimental_is_anonymous=is_anonymous)

    @def_function.function
    def lookup_table_func(k):
      return table.lookup(k)

    result = lookup_table_func(constant_op.constant([0, 1, -1]))
    self.assertAllEqual([b"brain", b"salad", b"n/a"], result)
    result = lookup_table_func(constant_op.constant([2, -1, 1]))
    self.assertAllEqual([b"surgery", b"n/a", b"salad"], result)

  def testTableCreatedInFunction(self, is_anonymous):
    if not context.executing_eagerly():
      self.skipTest("Only Eager mode test.")
    keys = constant_op.constant([0, 1, 2], dtypes.int32)
    values = constant_op.constant(["brain", "salad", "surgery"])

    @def_function.function
    def lookup_table_func(k):
      table = self.getHashTable()(
          lookup_ops.KeyValueTensorInitializer(keys, values),
          "n/a",
          experimental_is_anonymous=is_anonymous)
      return table.lookup(k)

    result = lookup_table_func(constant_op.constant([0, 1, -1]))
    self.assertAllEqual([b"brain", b"salad", b"n/a"], result)
    result = lookup_table_func(constant_op.constant([2, -1, 1]))
    self.assertAllEqual([b"surgery", b"n/a", b"salad"], result)

  def testTwoTablesInControlFlow(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    keys = constant_op.constant([1, 2, 3], dtypes.int32)
    values = constant_op.constant([5, 10, 15], dtypes.int32)

    def table_func1(x):
      table = self.getHashTable()(
          lookup_ops.KeyValueTensorInitializer(keys, values),
          -1,
          experimental_is_anonymous=is_anonymous)
      return table.lookup(x)

    elems = np.array([2, 4, 1], dtype=np.int32)
    result1 = map_fn.map_fn(table_func1, elems, dtype=dtypes.int32)

    def table_func2(x):
      table = self.getHashTable()(
          lookup_ops.KeyValueTensorInitializer(keys, values),
          -1,
          experimental_is_anonymous=is_anonymous)
      return table.lookup(x)

    elems = np.array([2, 4, 1], dtype=np.int32)
    result2 = map_fn.map_fn(table_func2, elems, dtype=dtypes.int32)

    self.evaluate(lookup_ops.tables_initializer())

    self.assertAllEqual([10, -1, 5], self.evaluate(result1))
    self.assertAllEqual([10, -1, 5], self.evaluate(result2))

  @test_util.enable_control_flow_v2
  def testLookupTableInWhileV2(self, is_anonymous):
    lookup = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(
            constant_op.constant([2, 5], dtype=dtypes.int64),
            constant_op.constant([-10.0, 1], dtype=dtypes.float32)),
        -1,
        experimental_is_anonymous=is_anonymous)

    beta = variables.Variable(1.0, trainable=True)

  @test_util.enable_control_flow_v2
  def testLookupTableInCondV2(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    lookup = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(
            constant_op.constant([2, 5], dtype=dtypes.int64),
            constant_op.constant([-10.0, 1], dtype=dtypes.float32)),
        -1,
        experimental_is_anonymous=is_anonymous)

    beta = variables.Variable(1.0, trainable=True)


  def testExportShapeInference(self, is_anonymous):
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(
            constant_op.constant([2, 5], dtype=dtypes.int64),
            constant_op.constant([-10.0, 1], dtype=dtypes.float32)),
        -1,
        experimental_is_anonymous=is_anonymous)
    actual_shapes = [t.shape for t in table.export()]
    inferred_shapes = []

    @def_function.function
    def f():
      for t in table.export():
        inferred_shapes.append(t.shape)

    f()
    self.assertLen(actual_shapes, 2)
    self.assertLen(inferred_shapes, 2)
    self.assertTrue(inferred_shapes[0].is_compatible_with(actual_shapes[0]))
    self.assertTrue(inferred_shapes[1].is_compatible_with(actual_shapes[1]))

  @test_util.run_v2_only
  def testSavedModelSaveRestore(self, is_anonymous):
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    root = autotrackable.AutoTrackable()

    default_value = -1
    keys = constant_op.constant([11, 12, 13], dtypes.int64)
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    root.table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_value,
        experimental_is_anonymous=is_anonymous)

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec((), dtypes.int64)])
    def lookup(key):
      return root.table.lookup(key)

    @def_function.function(input_signature=[])
    def size():
      return root.table.size()

    @def_function.function(input_signature=[])
    def is_ref_counting():
      return test_ops.is_resource_handle_ref_counting(
          root.table.resource_handle)

    root.lookup = lookup
    root.size = size
    root.is_ref_counting = is_ref_counting

    self.assertEqual(root.table.size(), 3)
    self.assertEqual(root.lookup(12), 1)
    self.assertEqual(root.lookup(10), -1)
    self.assertLen(root.table.export()[0], 3)
    self.assertEqual(root.is_ref_counting(), is_anonymous)

    saved_model_save.save(root, save_path)

    del root
    loaded = saved_model_load.load(save_path)
    self.assertEqual(loaded.size(), 3)
    self.assertEqual(loaded.lookup(12), 1)
    self.assertEqual(loaded.lookup(10), -1)
    self.assertEqual(loaded.is_ref_counting(), is_anonymous)


@parameterized.named_parameters(
    (f"_{is_anonymous}", is_anonymous) for is_anonymous in [False, True])
class KeyValueTensorInitializerTest(BaseLookupTableTest):

  def test_string(self, is_anonymous):
    init = lookup_ops.KeyValueTensorInitializer(
        ("brain", "salad", "surgery"), (0, 1, 2), dtypes.string, dtypes.int64)
    table = self.getHashTable()(
        init, default_value=-1, experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

  def test_multiple_tables(self, is_anonymous):
    with ops.name_scope("table_scope"):
      init1 = lookup_ops.KeyValueTensorInitializer(
          ("brain", "salad", "surgery"), (0, 1, 2), dtypes.string, dtypes.int64)
      table1 = self.getHashTable()(
          init1, default_value=-1, experimental_is_anonymous=is_anonymous)
      if not context.executing_eagerly():
        self.assertEqual("hash_table", table1.name)
        self.assertEqual("table_scope/hash_table",
                         table1.resource_handle.op.name)
      init2 = lookup_ops.KeyValueTensorInitializer(
          ("brain", "salad", "surgery"), (0, 1, 2), dtypes.string, dtypes.int64)
      table2 = self.getHashTable()(
          init2, default_value=-1, experimental_is_anonymous=is_anonymous)
      if not context.executing_eagerly():
        self.assertEqual("hash_table_1", table2.name)
        self.assertEqual("table_scope/hash_table_1",
                         table2.resource_handle.op.name)

  def test_int64(self, is_anonymous):
    init = lookup_ops.KeyValueTensorInitializer((42, 1, -1000), (0, 1, 2),
                                                dtypes.int64, dtypes.int64)
    table = self.getHashTable()(
        init, default_value=-1, experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

  def test_int32(self, is_anonymous):
    init = lookup_ops.KeyValueTensorInitializer((42, 1, -1000), (0, 1, 2),
                                                dtypes.int32, dtypes.int64)
    with self.assertRaises(errors_impl.OpError):
      table = self.getHashTable()(
          init, default_value=-1, experimental_is_anonymous=is_anonymous)
      self.initialize_table(table)


@parameterized.named_parameters(
    (f"_{is_anonymous}", is_anonymous) for is_anonymous in [False, True])
class StaticVocabularyTableTest(BaseLookupTableTest):

  def _createVocabFile(self, basename, values=("brain", "salad", "surgery")):
    vocabulary_file = os.path.join(self.get_temp_dir(), basename)
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(values) + "\n")
    return vocabulary_file

  def testInt32SparseTensor(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    input_indices = [[0, 0], [0, 1], [2, 0], [2, 2], [3, 0]]
    input_shape = [4, 4]
    sp_features = sparse_tensor.SparseTensor(
        constant_op.constant(input_indices, dtypes.int64),
        constant_op.constant([42, 1, 42, -1000, 11], dtypes.int32),
        constant_op.constant(input_shape, dtypes.int64))

    table = self.getVocabularyTable()(
        lookup_ops.KeyValueTensorInitializer((42, 1, -1000), (0, 1, 2),
                                             dtypes.int64, dtypes.int64),
        1,
        lookup_key_dtype=dtypes.int32,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    sp_ids = table.lookup(sp_features)

    self.assertAllEqual([5], sp_ids.values._shape_as_list())

    sp_ids_ind, sp_ids_val, sp_ids_shape = self.evaluate(
        [sp_ids.indices, sp_ids.values, sp_ids.dense_shape])

    self.assertAllEqual(input_indices, sp_ids_ind)
    self.assertAllEqual([0, 1, 0, 2, 3], sp_ids_val)
    self.assertAllEqual(input_shape, sp_ids_shape)

  def testInt32RaggedTensor(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    input_row_splits = [0, 2, 4, 5]
    ragged_features = ragged_tensor.RaggedTensor.from_row_splits(
        constant_op.constant([42, 1, 42, -1000, 11], dtypes.int32),
        constant_op.constant(input_row_splits, dtypes.int64))

    table = self.getVocabularyTable()(
        lookup_ops.KeyValueTensorInitializer((42, 1, -1000), (0, 1, 2),
                                             dtypes.int64, dtypes.int64),
        1,
        lookup_key_dtype=dtypes.int32,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    ragged_ids = table.lookup(ragged_features)

    self.assertAllEqual([5], ragged_ids.values._shape_as_list())

    ragged_ids_val, ragged_ids_row_splits = self.evaluate(
        [ragged_ids.values, ragged_ids.row_splits])

    self.assertAllEqual([0, 1, 0, 2, 3], ragged_ids_val)
    self.assertAllEqual(input_row_splits, ragged_ids_row_splits)

  def testInt64SparseTensor(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    input_indices = [[0, 0], [0, 1], [2, 0], [2, 2], [3, 0]]
    input_shape = [4, 4]
    sp_features = sparse_tensor.SparseTensor(
        constant_op.constant(input_indices, dtypes.int64),
        constant_op.constant([42, 1, 42, -1000, 11], dtypes.int64),
        constant_op.constant(input_shape, dtypes.int64))

    table = self.getVocabularyTable()(
        lookup_ops.KeyValueTensorInitializer((42, 1, -1000), (0, 1, 2),
                                             dtypes.int64, dtypes.int64),
        1,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    sp_ids = table.lookup(sp_features)

    self.assertAllEqual([5], sp_ids.values._shape_as_list())

    sp_ids_ind, sp_ids_val, sp_ids_shape = self.evaluate(
        [sp_ids.indices, sp_ids.values, sp_ids.dense_shape])

    self.assertAllEqual(input_indices, sp_ids_ind)
    self.assertAllEqual([0, 1, 0, 2, 3], sp_ids_val)
    self.assertAllEqual(input_shape, sp_ids_shape)

  def testInt64RaggedTensor(self, is_anonymous):
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    input_row_splits = [0, 2, 4, 5]
    ragged_features = ragged_tensor.RaggedTensor.from_row_splits(
        constant_op.constant([42, 1, 42, -1000, 11], dtypes.int64),
        constant_op.constant(input_row_splits, dtypes.int64))

    table = self.getVocabularyTable()(
        lookup_ops.KeyValueTensorInitializer((42, 1, -1000), (0, 1, 2),
                                             dtypes.int64, dtypes.int64),
        1,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    ragged_ids = table.lookup(ragged_features)

    self.assertAllEqual([5], ragged_ids.values._shape_as_list())

    ragged_ids_val, ragged_ids_row_splits = self.evaluate(
        [ragged_ids.values, ragged_ids.row_splits])

    self.assertAllEqual([0, 1, 0, 2, 3], ragged_ids_val)
    self.assertAllEqual(input_row_splits, ragged_ids_row_splits)


class IndexToStringTableFromFileTest(test.TestCase):

  def _createVocabFile(self, basename, values=("brain", "salad", "surgery")):
    vocabulary_file = os.path.join(self.get_temp_dir(), basename)
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(values) + "\n")
    return vocabulary_file

  def testInt32SparseTensor(self):
    input_indices = [[0, 0], [0, 1], [2, 0], [2, 2], [3, 0]]
    input_shape = [4, 4]
    sp_features = sparse_tensor.SparseTensor(
        constant_op.constant(input_indices, dtypes.int64),
        constant_op.constant([42, 1, 42, -1000, 11], dtypes.int32),
        constant_op.constant(input_shape, dtypes.int64))

    table = lookup_ops.IdTableWithHashBuckets(
        lookup_ops.StaticHashTable(
            lookup_ops.KeyValueTensorInitializer(
                (42, 1, -1000), (0, 1, 2), dtypes.int64, dtypes.int64), -1),
        1,
        key_dtype=dtypes.int32)
    self.evaluate(table.initializer)

    sp_ids = table.lookup(sp_features)

    self.assertAllEqual([5], sp_ids.values._shape_as_list())

    sp_ids_ind, sp_ids_val, sp_ids_shape = self.evaluate(
        [sp_ids.indices, sp_ids.values, sp_ids.dense_shape])

    self.assertAllEqual(input_indices, sp_ids_ind)
    self.assertAllEqual([0, 1, 0, 2, 3], sp_ids_val)
    self.assertAllEqual(input_shape, sp_ids_shape)

  def testInt32RaggedTensor(self):
    input_row_splits = [0, 2, 4, 5]
    ragged_features = ragged_tensor.RaggedTensor.from_row_splits(
        constant_op.constant([42, 1, 42, -1000, 11], dtypes.int32),
        constant_op.constant(input_row_splits, dtypes.int32))

    table = lookup_ops.IdTableWithHashBuckets(
        lookup_ops.StaticHashTable(
            lookup_ops.KeyValueTensorInitializer(
                (42, 1, -1000), (0, 1, 2), dtypes.int64, dtypes.int64), -1),
        1,
        key_dtype=dtypes.int32)
    self.evaluate(table.initializer)

    ragged_ids = table.lookup(ragged_features)

    self.assertAllEqual([5], ragged_ids.values._shape_as_list())

    ragged_ids_val, ragged_ids_row_splits = self.evaluate(
        [ragged_ids.values, ragged_ids.row_splits])

    self.assertAllEqual([0, 1, 0, 2, 3], ragged_ids_val)
    self.assertAllEqual(input_row_splits, ragged_ids_row_splits)

  def testInt64SparseTensor(self):
    input_indices = [[0, 0], [0, 1], [2, 0], [2, 2], [3, 0]]
    input_shape = [4, 4]
    sp_features = sparse_tensor.SparseTensor(
        constant_op.constant(input_indices, dtypes.int64),
        constant_op.constant([42, 1, 42, -1000, 11], dtypes.int64),
        constant_op.constant(input_shape, dtypes.int64))

    table = lookup_ops.IdTableWithHashBuckets(
        lookup_ops.StaticHashTable(
            lookup_ops.KeyValueTensorInitializer(
                (42, 1, -1000), (0, 1, 2), dtypes.int64, dtypes.int64), -1),
        1,
        key_dtype=dtypes.int64)
    self.evaluate(table.initializer)

    sp_ids = table.lookup(sp_features)

    self.assertAllEqual([5], sp_ids.values._shape_as_list())

    sp_ids_ind, sp_ids_val, sp_ids_shape = self.evaluate(
        [sp_ids.indices, sp_ids.values, sp_ids.dense_shape])

    self.assertAllEqual(input_indices, sp_ids_ind)
    self.assertAllEqual([0, 1, 0, 2, 3], sp_ids_val)
    self.assertAllEqual(input_shape, sp_ids_shape)

  def testInt64RaggedTensor(self):
    input_row_splits = [0, 2, 4, 5]
    ragged_features = ragged_tensor.RaggedTensor.from_row_splits(
        constant_op.constant([42, 1, 42, -1000, 11], dtypes.int64),
        constant_op.constant(input_row_splits, dtypes.int64))

    table = lookup_ops.IdTableWithHashBuckets(
        lookup_ops.StaticHashTable(
            lookup_ops.KeyValueTensorInitializer(
                (42, 1, -1000), (0, 1, 2), dtypes.int64, dtypes.int64), -1),
        1,
        key_dtype=dtypes.int64)
    self.evaluate(table.initializer)

    ragged_ids = table.lookup(ragged_features)

    self.assertAllEqual([5], ragged_ids.values._shape_as_list())

    ragged_ids_val, ragged_ids_row_splits = self.evaluate(
        [ragged_ids.values, ragged_ids.row_splits])

    self.assertAllEqual([0, 1, 0, 2, 3], ragged_ids_val)
    self.assertAllEqual(input_row_splits, ragged_ids_row_splits)

  
class MutableHashTableBenchmark(test.Benchmark):

  def _create_table(self):
    return lookup_ops.MutableHashTable(dtypes.int64, dtypes.float32, 0.0)

  def benchmark_single_repeated_scalar_insert_scalar(self):
    table = self._create_table()
    value = variables.Variable(1.0)
    insert = table.insert(0, value)
    size = table.size()
    with session.Session() as sess:
      sess.run(value.initializer)
      self.run_op_benchmark(sess, insert, burn_iters=10, min_iters=10000)
      assert sess.run(size) == 1

  def benchmark_many_repeated_scalar_insert_scalar(self):
    table = self._create_table()
    c = dataset_ops.make_one_shot_iterator(counter.Counter()).get_next()
    value = variables.Variable(1.0)
    insert = table.insert(c, value)
    size = table.size()
    with session.Session() as sess:
      sess.run(value.initializer)
      self.run_op_benchmark(sess, insert, burn_iters=10, min_iters=10000)
      assert sess.run(size) >= 10000

  def benchmark_single_repeated_batch_32_insert_scalar(self):
    table = self._create_table()
    value = variables.Variable([1.0] * 32)
    insert = table.insert(list(range(32)), value)
    size = table.size()
    with session.Session() as sess:
      sess.run(value.initializer)
      self.run_op_benchmark(sess, insert, burn_iters=10, min_iters=1000)
      assert sess.run(size) == 32

  def benchmark_many_repeated_batch_32_insert_scalar(self):
    table = self._create_table()
    c = dataset_ops.make_one_shot_iterator(counter.Counter()).get_next()
    value = variables.Variable([1.0] * 32)
    insert = table.insert(32 * c + list(range(32)), value)
    size = table.size()
    with session.Session() as sess:
      sess.run(value.initializer)
      self.run_op_benchmark(sess, insert, burn_iters=10, min_iters=1000)
      assert sess.run(size) >= 1000 * 32


class DenseHashTableBenchmark(MutableHashTableBenchmark):

  def _create_table(self):
    return lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.float32,
        default_value=0.0,
        empty_key=-1,
        deleted_key=-2)


if __name__ == "__main__":
  test.main()
