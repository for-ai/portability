# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for pfor and for_loop."""
# pylint: disable=g-direct-tensorflow-import

import functools
import sys
import time

from absl.testing import parameterized
import numpy as np

from google.protobuf import text_format

from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import importer
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import cond_v2
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_v2_toggles
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_list_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients as gradient_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.parallel_for import control_flow_ops as pfor_control_flow_ops
from tensorflow.python.ops.parallel_for.test_util import PForTestCase
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.platform import test
from tensorflow.python.util import nest


class VariableTest(PForTestCase):

    def test_create_variable_once(self):
        x = array_ops.ones(shape=(3, 2, 2), dtype=dtypes.float32)
        y = array_ops.ones(shape=(2, 3), dtype=dtypes.float32)
        a_var = []

        def f(z):
            if not a_var:
                a_var.append(variables.Variable(lambda: y, name="a"))
            return math_ops.matmul(z, a_var[0] / 16)

        pfor_control_flow_ops.vectorized_map(f, x)

    @test_util.run_v2_only
    def test_create_variable_repeated(self):
        x = array_ops.ones(shape=(3, 2, 2), dtype=dtypes.float32)
        y = array_ops.ones(shape=(2, 3), dtype=dtypes.float32)

        def f(z):
            a_var = variables.Variable(lambda: y, name="a") / 4
            return math_ops.matmul(z, a_var / 16)

        # Note that this error is only raised under v2 behavior.
        with self.assertRaisesRegex(
                ValueError, "singleton tf.Variable.*on the first call"):
            pfor_control_flow_ops.vectorized_map(f, x)

    @test_util.run_all_in_graph_and_eager_modes
    def test_variable_shape(self):
        v = resource_variable_ops.ResourceVariable([1, 2])

        def loop_fn(_):
            return resource_variable_ops.variable_shape(v.handle)

        self._test_loop_fn(loop_fn, 2)

    @test_util.run_all_in_graph_and_eager_modes
    def test_variable_input(self):
        v = resource_variable_ops.ResourceVariable([1, 2])
        self.evaluate(v.initializer)

        def loop_fn(x):
            return x + 1

        result = pfor_control_flow_ops.vectorized_map(loop_fn, v)
        expected_result = [2, 3]
        self.assertAllEqual(result, expected_result)


if __name__ == "__main__":
    test.main()
