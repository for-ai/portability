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

import importlib

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.distributions import laplace as laplace_lib
from tensorflow.python.platform import test

from tensorflow.python.platform import tf_logging
from ..utils.timer_wrapper import tensorflow_op_timer


def try_import(name):  # pylint: disable=invalid-name
  module = None
  try:
    module = importlib.import_module(name)
  except ImportError as e:
    tf_logging.warning("Could not import %s: %s" % (name, str(e)))
  return module


stats = try_import("scipy.stats")


@test_util.run_all_in_graph_and_eager_modes
class LaplaceTest(test.TestCase):

  def testLaplaceLogSurvivalFunction(self):
    batch_size = 6
    loc = constant_op.constant([2.0] * batch_size)
    scale = constant_op.constant([3.0] * batch_size)
    loc_v = 2.0
    scale_v = 3.0
    x = np.array([-2.5, 2.5, -4.0, 0.1, 1.0, 2.0], dtype=np.float32)

    laplace = laplace_lib.Laplace(loc=loc, scale=scale)
    with tensorflow_op_timer():
        sf = laplace.log_survival_function(x)
    self.assertEqual(sf.get_shape(), (6,))
    if not stats:
      return
    expected_sf = stats.laplace.logsf(x, loc_v, scale=scale_v)
    self.assertAllClose(self.evaluate(sf), expected_sf)

  
if __name__ == "__main__":
  test.main()
