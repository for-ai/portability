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
from ..utils.timer_wrapper import tensorflow_op_timer


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
class KeyValueTensorInitializerTest(BaseLookupTableTest):

  def test_string(self, is_anonymous):
    timer = tensorflow_op_timer()
    with timer:
      init = lookup_ops.KeyValueTensorInitializer(
        ("brain", "salad", "surgery"), (0, 1, 2), dtypes.string, dtypes.int64)
      timer.gen.send(init)
    table = self.getHashTable()(
        init, default_value=-1, experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

  def test_multiple_tables(self, is_anonymous):
    with ops.name_scope("table_scope"):
      timer = tensorflow_op_timer()
      with timer:
        init1 = lookup_ops.KeyValueTensorInitializer(
          ("brain", "salad", "surgery"), (0, 1, 2), dtypes.string, dtypes.int64)
        timer.gen.send(init1)
      table1 = self.getHashTable()(
          init1, default_value=-1, experimental_is_anonymous=is_anonymous)
      if not context.executing_eagerly():
        self.assertEqual("hash_table", table1.name)
        self.assertEqual("table_scope/hash_table",
                         table1.resource_handle.op.name)
      timer = tensorflow_op_timer()
      with timer:
        init2 = lookup_ops.KeyValueTensorInitializer(
          ("brain", "salad", "surgery"), (0, 1, 2), dtypes.string, dtypes.int64)
        timer.gen.send(init2)
      table2 = self.getHashTable()(
          init2, default_value=-1, experimental_is_anonymous=is_anonymous)
      if not context.executing_eagerly():
        self.assertEqual("hash_table_1", table2.name)
        self.assertEqual("table_scope/hash_table_1",
                         table2.resource_handle.op.name)

  def test_int64(self, is_anonymous):
    timer = tensorflow_op_timer()
    with timer:
      init = lookup_ops.KeyValueTensorInitializer((42, 1, -1000), (0, 1, 2),
                                                dtypes.int64, dtypes.int64)
      timer.gen.send(init)
    table = self.getHashTable()(
        init, default_value=-1, experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

  def test_int32(self, is_anonymous):
    timer = tensorflow_op_timer()
    with timer:
      init = lookup_ops.KeyValueTensorInitializer((42, 1, -1000), (0, 1, 2),
                                                dtypes.int32, dtypes.int64)
      timer.gen.send(init)
    with self.assertRaises(errors_impl.OpError):
      table = self.getHashTable()(
          init, default_value=-1, experimental_is_anonymous=is_anonymous)
      self.initialize_table(table)



if __name__ == "__main__":
  test.main()
