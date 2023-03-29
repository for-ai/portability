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
"""Tests for tensorflow.python.client.session.Session."""
import collections
import os
import random
import sys
import threading
import time
import warnings

import numpy as np
import six

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.lib.core import error_codes_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as framework_device_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import importer
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.framework import versions
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_control_flow_ops
# Import gradients to resolve circular imports
from tensorflow.python.ops import gradients  # pylint: disable=unused-import
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
# Import resource_variable_ops for the variables-to-tensor implicit conversion.
from tensorflow.python.ops import resource_variable_ops  # pylint: disable=unused-import
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat
from ..tensorflow_test import device_context

try:
    import attr  # pylint:disable=g-import-not-at-top
except ImportError:
    attr = None

try:
    from frozendict import frozendict  # pylint:disable=g-import-not-at-top
except ImportError:
    frozendict = dict  # pylint:disable=invalid-name

defaultdict = collections.defaultdict  # pylint:disable=invalid-name


@test_util.with_eager_op_as_function
class SessionTest(test_util.TensorFlowTestCase):

    def setUp(self):
        super(SessionTest, self).setUp()
        warnings.simplefilter('always')

    def testUseExistingGraph(self):
        with ops.Graph().as_default() as g, ops.device('/cpu:0'):
            a = constant_op.constant(6.0, shape=[1, 1])
            b = constant_op.constant(7.0, shape=[1, 1])
            c = math_ops.matmul(a, b, name='matmul')
        with session.Session(graph=g):
            with device_context():
                result = c.eval()
                self.assertAllEqual(result, [[42.0]])

    def testUseDefaultGraph(self):
        with ops.Graph().as_default(), ops.device('/cpu:0'):
            a = constant_op.constant(6.0, shape=[1, 1])
            b = constant_op.constant(7.0, shape=[1, 1])
            c = math_ops.matmul(a, b, name='matmul')
            with session.Session():
                with device_context():
                    result = c.eval()
                    self.assertAllEqual(result, [[42.0]])

    def testCreate(self):
        with session.Session():
            with device_context():
                inp = constant_op.constant(10.0, shape=[2, 3], name='W1')
                copy = array_ops.identity(inp)
                # Test with feed.
                # TODO(mrry): Investigate why order='F' didn't work.
                arr = np.asarray([[0, 1, 2], [3, 4, 5]],
                                dtype=np.float32, order='C')
                copy_val = copy.eval({'W1:0': arr})
                self.assertAllEqual(arr, copy_val)
                # Test without feed.
                copy_val = copy.eval()
                self.assertAllEqual(
                    np.asarray(
                        [[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]], dtype=np.float32),
                    copy_val)


if __name__ == '__main__':
    googletest.main()
