# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.test_util."""

import collections
import copy
import random
import threading
import unittest
import weakref

from absl.testing import parameterized
import numpy as np

from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python import pywrap_sanitizers
from tensorflow.python.compat import compat
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_ops  # pylint: disable=unused-import
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest


class TestUtilTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  @test_util.run_in_graph_and_eager_modes
  def testAssertAllGreaterLessEqual(self):
    x = constant_op.constant([100.0, 110.0, 120.0], dtype=dtypes.float32)
    y = constant_op.constant([10.0] * 3, dtype=dtypes.float32)
    z = math_ops.add(x, y)

    self.assertAllEqual([110.0, 120.0, 130.0], z)

    self.assertAllGreaterEqual(x, 95.0)
    self.assertAllLessEqual(x, 125.0)

    with self.assertRaises(AssertionError):
      self.assertAllGreaterEqual(x, 105.0)
    with self.assertRaises(AssertionError):
      self.assertAllGreaterEqual(x, 125.0)

    with self.assertRaises(AssertionError):
      self.assertAllLessEqual(x, 115.0)
    with self.assertRaises(AssertionError):
      self.assertAllLessEqual(x, 95.0)

if __name__ == "__main__":
  googletest.main()
