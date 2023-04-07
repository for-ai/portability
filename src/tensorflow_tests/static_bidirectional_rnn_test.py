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
"""Tests for RNN cells."""

import itertools
import os

from absl.testing import parameterized
import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_rnn_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import save
from tensorflow.python.trackable import autotrackable
from tensorflow.python.util import nest


class BidirectionalRNNTest(test.TestCase):

    def setUp(self):
        self._seed = 23489
        np.random.seed(self._seed)

    def _createBidirectionalRNN(self, use_shape, use_sequence_length, scope=None):
        num_units = 3
        input_size = 5
        batch_size = 2
        max_length = 8

        initializer = init_ops.random_uniform_initializer(
            -0.01, 0.01, seed=self._seed)
        sequence_length = array_ops.placeholder(
            dtypes.int64) if use_sequence_length else None
        cell_fw = rnn_cell.LSTMCell(
            num_units, input_size, initializer=initializer, state_is_tuple=False)
        cell_bw = rnn_cell.LSTMCell(
            num_units, input_size, initializer=initializer, state_is_tuple=False)
        inputs = max_length * [
            array_ops.placeholder(
                dtypes.float32,
                shape=(batch_size, input_size) if use_shape else (None, input_size))
        ]
        outputs, state_fw, state_bw = rnn.static_bidirectional_rnn(
            cell_fw,
            cell_bw,
            inputs,
            dtype=dtypes.float32,
            sequence_length=sequence_length,
            scope=scope)
        self.assertEqual(len(outputs), len(inputs))
        for out in outputs:
            self.assertEqual(out.get_shape().as_list(),
                             [batch_size if use_shape else None, 2 * num_units])

        input_value = np.random.randn(batch_size, input_size)
        outputs = array_ops.stack(outputs)

        return input_value, inputs, outputs, state_fw, state_bw, sequence_length

    def _testBidirectionalRNN(self, use_shape):
        with self.session(graph=ops.Graph()) as sess:
            input_value, inputs, outputs, state_fw, state_bw, sequence_length = (
                self._createBidirectionalRNN(use_shape, True))
            variables_lib.global_variables_initializer().run()
            # Run with pre-specified sequence length of 2, 3
            out, s_fw, s_bw = sess.run(
                [outputs, state_fw, state_bw],
                feed_dict={
                    inputs[0]: input_value,
                    sequence_length: [2, 3]
                })

            # Since the forward and backward LSTM cells were initialized with the
            # same parameters, the forward and backward output has to be the same,
            # but reversed in time. The format is output[time][batch][depth], and
            # due to depth concatenation (as num_units=3 for both RNNs):
            # - forward output:  out[][][depth] for 0 <= depth < 3
            # - backward output: out[][][depth] for 4 <= depth < 6
            #
            # First sequence in batch is length=2
            # Check that the time=0 forward output is equal to time=1 backward output
            self.assertAllClose(out[0][0][0], out[1][0][3])
            self.assertAllClose(out[0][0][1], out[1][0][4])
            self.assertAllClose(out[0][0][2], out[1][0][5])
            # Check that the time=1 forward output is equal to time=0 backward output
            self.assertAllClose(out[1][0][0], out[0][0][3])
            self.assertAllClose(out[1][0][1], out[0][0][4])
            self.assertAllClose(out[1][0][2], out[0][0][5])

            # Second sequence in batch is length=3
            # Check that the time=0 forward output is equal to time=2 backward output
            self.assertAllClose(out[0][1][0], out[2][1][3])
            self.assertAllClose(out[0][1][1], out[2][1][4])
            self.assertAllClose(out[0][1][2], out[2][1][5])
            # Check that the time=1 forward output is equal to time=1 backward output
            self.assertAllClose(out[1][1][0], out[1][1][3])
            self.assertAllClose(out[1][1][1], out[1][1][4])
            self.assertAllClose(out[1][1][2], out[1][1][5])
            # Check that the time=2 forward output is equal to time=0 backward output
            self.assertAllClose(out[2][1][0], out[0][1][3])
            self.assertAllClose(out[2][1][1], out[0][1][4])
            self.assertAllClose(out[2][1][2], out[0][1][5])
            # Via the reasoning above, the forward and backward final state should be
            # exactly the same
            self.assertAllClose(s_fw, s_bw)

    def _testBidirectionalRNNWithoutSequenceLength(self, use_shape):
        with self.session(graph=ops.Graph()) as sess:
            input_value, inputs, outputs, state_fw, state_bw, _ = (
                self._createBidirectionalRNN(use_shape, False))
            variables_lib.global_variables_initializer().run()
            out, s_fw, s_bw = sess.run(
                [outputs, state_fw, state_bw], feed_dict={
                    inputs[0]: input_value
                })

            # Since the forward and backward LSTM cells were initialized with the
            # same parameters, the forward and backward output has to be the same,
            # but reversed in time. The format is output[time][batch][depth], and
            # due to depth concatenation (as num_units=3 for both RNNs):
            # - forward output:  out[][][depth] for 0 <= depth < 3
            # - backward output: out[][][depth] for 4 <= depth < 6
            #
            # Both sequences in batch are length=8.  Check that the time=i
            # forward output is equal to time=8-1-i backward output
            for i in range(8):
                self.assertAllClose(out[i][0][0:3], out[8 - 1 - i][0][3:6])
                self.assertAllClose(out[i][1][0:3], out[8 - 1 - i][1][3:6])
            # Via the reasoning above, the forward and backward final state should be
            # exactly the same
            self.assertAllClose(s_fw, s_bw)

    @test_util.run_v1_only("b/124229375")
    def testBidirectionalRNN(self):
        self._testBidirectionalRNN(use_shape=False)
        self._testBidirectionalRNN(use_shape=True)


if __name__ == "__main__":
    test.main()
