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
"""Tests for supervisor.py."""

import glob
import os
import shutil
import time
import uuid


from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.summary import summary
from tensorflow.python.summary import summary_iterator
from tensorflow.python.summary.writer import writer
from tensorflow.python.training import input as input_lib
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import server_lib
from tensorflow.python.training import session_manager as session_manager_lib
from tensorflow.python.training import supervisor


def _summary_iterator(test_dir):
  """Reads events from test_dir/events.

  Args:
    test_dir: Name of the test directory.

  Returns:
    A summary_iterator
  """
  event_paths = sorted(glob.glob(os.path.join(test_dir, "event*")))
  return summary_iterator.summary_iterator(event_paths[-1])


class SupervisorTest(test.TestCase):

  def _test_dir(self, test_name):
    test_dir = os.path.join(self.get_temp_dir(), test_name)
    if os.path.exists(test_dir):
      shutil.rmtree(test_dir)
    return test_dir

  def _wait_for_glob(self, pattern, timeout_secs, for_checkpoint=True):
    """Wait for a checkpoint file to appear.

    Args:
      pattern: A string.
      timeout_secs: How long to wait for in seconds.
      for_checkpoint: whether we're globbing for checkpoints.
    """
    end_time = time.time() + timeout_secs
    while time.time() < end_time:
      if for_checkpoint:
        if checkpoint_management.checkpoint_exists(pattern):
          return
      else:
        if len(gfile.Glob(pattern)) >= 1:
          return
      time.sleep(0.05)
    self.assertFalse(True, "Glob never matched any file: %s" % pattern)
    

  def testManagedSession(self):
    logdir = self._test_dir("managed_session")
    with ops.Graph().as_default():
      my_op = constant_op.constant(1.0)
      sv = supervisor.Supervisor(logdir=logdir)
      with sv.managed_session(""):
        for _ in range(10):
          self.evaluate(my_op)
      # Supervisor has been stopped.
      self.assertTrue(sv.should_stop())

  
  def _csv_data(self, logdir):
    # Create a small data file with 3 CSV records.
    data_path = os.path.join(logdir, "data.csv")
    with open(data_path, "w") as f:
      f.write("1,2,3\n")
      f.write("4,5,6\n")
      f.write("7,8,9\n")
    return data_path

if __name__ == "__main__":
  test.main()
