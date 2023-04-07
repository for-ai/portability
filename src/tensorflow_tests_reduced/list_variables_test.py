# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for checkpoints tools."""

import os
import pathlib
import time

import numpy as np

import tensorflow as tf

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import saver as saver_lib
from ..utils.timer_wrapper import tensorflow_op_timer


def _create_checkpoints(sess, checkpoint_dir):

  checkpoint_prefix = os.path.join(checkpoint_dir, "model")
  checkpoint_state_name = "checkpoint"
  v1 = variable_scope.get_variable("var1", [1, 10])
  v2 = variable_scope.get_variable("var2", [10, 10])
  v3 = variable_scope.get_variable("var3", [100, 100])
  with variable_scope.variable_scope("useful_scope"):
    v4 = variable_scope.get_variable("var4", [9, 9])
  sess.run(variables.global_variables_initializer())
  v1_value, v2_value, v3_value, v4_value = sess.run([v1, v2, v3, v4])
  saver = saver_lib.Saver()
  saver.save(
      sess,
      checkpoint_prefix,
      global_step=0,
      latest_filename=checkpoint_state_name)
  return v1_value, v2_value, v3_value, v4_value


def _create_partition_checkpoints(sess, checkpoint_dir):
  checkpoint_prefix = os.path.join(checkpoint_dir, "model")
  checkpoint_state_name = "checkpoint"
  with variable_scope.variable_scope("scope"):
    v1 = variable_scope.get_variable(
        name="var1",
        shape=[100, 100],
        initializer=init_ops.truncated_normal_initializer(0.5),
        partitioner=partitioned_variables.min_max_variable_partitioner(
            max_partitions=5, axis=0, min_slice_size=8 << 10))
  sess.run(variables.global_variables_initializer())
  v1_value = sess.run(v1._get_variable_list())
  saver = saver_lib.Saver()
  saver.save(
      sess,
      checkpoint_prefix,
      global_step=0,
      latest_filename=checkpoint_state_name)
  return v1_value

@test_util.run_all_in_deprecated_graph_mode_only
class CheckpointsTest(test.TestCase):

  # def setUp(self):
  #     super(CheckpointsTest, self).setUp()
      # tf.compat.v1.disable_eager_execution() # Dung: have to add this 
  def testGetAllVariables(self):
    checkpoint_dir = self.get_temp_dir()
    with self.cached_session() as session:
      _create_checkpoints(session, checkpoint_dir)
    with tensorflow_op_timer():
      test = checkpoint_utils.list_variables(checkpoint_dir)
    self.assertEqual(
        checkpoint_utils.list_variables(checkpoint_dir),
        [("useful_scope/var4", [9, 9]), ("var1", [1, 10]), ("var2", [10, 10]),
         ("var3", [100, 100])])

  def testFSPath(self):
    checkpoint_dir = pathlib.Path(self.get_temp_dir())
    with self.cached_session() as session:
      v1, v2, v3, v4 = _create_checkpoints(session, checkpoint_dir)  # pylint: disable=unused-variable

    reader = checkpoint_utils.load_checkpoint(checkpoint_dir)
    self.assertAllEqual(reader.get_tensor("var1"), v1)

    self.assertAllEqual(
        checkpoint_utils.load_variable(checkpoint_dir, "var1"), v1)
    with tensorflow_op_timer():
      test = checkpoint_utils.list_variables(checkpoint_dir)
    self.assertEqual(
        checkpoint_utils.list_variables(checkpoint_dir),
        [("useful_scope/var4", [9, 9]), ("var1", [1, 10]), ("var2", [10, 10]),
         ("var3", [100, 100])])


  

if __name__ == "__main__":
  test.main()
