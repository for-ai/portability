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
from tensorflow.python.ops import   array_ops
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





class RawRNNTest(test.TestCase):

  def setUp(self):
    self._seed = 23489
    np.random.seed(self._seed)

  @test_util.run_v1_only("b/124229375")
  def _testRawRNN(self, max_time):
    with self.session(graph=ops.Graph()) as sess:
      batch_size = 16
      input_depth = 4
      num_units = 3

      inputs = array_ops.placeholder(
          shape=(max_time, batch_size, input_depth), dtype=dtypes.float32)
      sequence_length = array_ops.placeholder(
          shape=(batch_size,), dtype=dtypes.int32)
      inputs_ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, size=array_ops.shape(inputs)[0])
      inputs_ta = inputs_ta.unstack(inputs)

      cell = rnn_cell.LSTMCell(num_units, state_is_tuple=True)

      def loop_fn(time_, cell_output, cell_state, unused_loop_state):
        emit_output = cell_output  # == None for time == 0
        if cell_output is None:  # time == 0
          next_state = cell.zero_state(batch_size, dtypes.float32)
        else:
          next_state = cell_state  # copy state through
        elements_finished = (time_ >= sequence_length)
        finished = math_ops.reduce_all(elements_finished)
        # For the very final iteration, we must emit a dummy input
        next_input = control_flow_ops.cond(
            finished,
            lambda: array_ops.zeros([batch_size, input_depth], dtype=dtypes.float32),
            lambda: inputs_ta.read(time_))
        return (elements_finished, next_input, next_state, emit_output, None)

      reuse_scope = variable_scope.get_variable_scope()

      outputs_ta, final_state, _ = rnn.raw_rnn(cell, loop_fn, scope=reuse_scope)
      outputs = outputs_ta.stack()

      reuse_scope.reuse_variables()
      outputs_dynamic_rnn, final_state_dynamic_rnn = rnn.dynamic_rnn(
          cell,
          inputs,
          time_major=True,
          dtype=dtypes.float32,
          sequence_length=sequence_length,
          scope=reuse_scope)

      variables = variables_lib.trainable_variables()
      gradients = gradients_impl.gradients([outputs, final_state],
                                           [inputs] + variables)
      gradients_dynamic_rnn = gradients_impl.gradients(
          [outputs_dynamic_rnn, final_state_dynamic_rnn], [inputs] + variables)

      variables_lib.global_variables_initializer().run()

      rand_input = np.random.randn(max_time, batch_size, input_depth)
      if max_time == 0:
        rand_seq_len = np.zeros(batch_size)
      else:
        rand_seq_len = np.random.randint(max_time, size=batch_size)

      # To ensure same output lengths for dynamic_rnn and raw_rnn
      rand_seq_len[0] = max_time

      (outputs_val, outputs_dynamic_rnn_val, final_state_val,
       final_state_dynamic_rnn_val) = sess.run(
           [outputs, outputs_dynamic_rnn, final_state, final_state_dynamic_rnn],
           feed_dict={
               inputs: rand_input,
               sequence_length: rand_seq_len
           })

      self.assertAllClose(outputs_dynamic_rnn_val, outputs_val)
      self.assertAllClose(final_state_dynamic_rnn_val, final_state_val)

      # NOTE: Because with 0 time steps, raw_rnn does not have shape
      # information about the input, it is impossible to perform
      # gradients comparisons as the gradients eval will fail.  So
      # this case skips the gradients test.
      if max_time > 0:
        self.assertEqual(len(gradients), len(gradients_dynamic_rnn))
        gradients_val = sess.run(
            gradients,
            feed_dict={
                inputs: rand_input,
                sequence_length: rand_seq_len
            })
        gradients_dynamic_rnn_val = sess.run(
            gradients_dynamic_rnn,
            feed_dict={
                inputs: rand_input,
                sequence_length: rand_seq_len
            })
        self.assertEqual(len(gradients_val), len(gradients_dynamic_rnn_val))
        input_gradients_val = gradients_val[0]
        input_gradients_dynamic_rnn_val = gradients_dynamic_rnn_val[0]
        self.assertAllClose(input_gradients_val,
                            input_gradients_dynamic_rnn_val)
        for i in range(1, len(gradients_val)):
          self.assertAllClose(gradients_dynamic_rnn_val[i], gradients_val[i])

  @test_util.run_v1_only("b/124229375")
  def testRawRNNZeroLength(self):
    # NOTE: Because with 0 time steps, raw_rnn does not have shape
    # information about the input, it is impossible to perform
    # gradients comparisons as the gradients eval will fail.  So this
    # case skips the gradients test.
    self._testRawRNN(max_time=0)

  def testRawRNN(self):
    self._testRawRNN(max_time=10)

  @test_util.run_v1_only("b/124229375")
  def testLoopState(self):
    with self.session(graph=ops.Graph()):
      max_time = 10
      batch_size = 16
      input_depth = 4
      num_units = 3

      inputs = np.random.randn(max_time, batch_size, input_depth)
      inputs_ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, size=array_ops.shape(inputs)[0])
      inputs_ta = inputs_ta.unstack(inputs)

      cell = rnn_cell.LSTMCell(num_units, state_is_tuple=True)

      def loop_fn(time_, cell_output, cell_state, loop_state):
        if cell_output is None:
          loop_state = constant_op.constant([0])
          next_state = cell.zero_state(batch_size, dtypes.float32)
        else:
          loop_state = array_ops.stack([array_ops.squeeze(loop_state) + 1])
          next_state = cell_state
        emit_output = cell_output  # == None for time == 0
        elements_finished = array_ops.tile([time_ >= max_time], [batch_size])
        finished = math_ops.reduce_all(elements_finished)
        # For the very final iteration, we must emit a dummy input
        next_input = control_flow_ops.cond(
            finished,
            lambda: array_ops.zeros([batch_size, input_depth], dtype=dtypes.float32),
            lambda: inputs_ta.read(time_))
        return (elements_finished, next_input, next_state, emit_output,
                loop_state)

      r = rnn.raw_rnn(cell, loop_fn)
      loop_state = r[-1]
      self.assertEqual([10], self.evaluate(loop_state))

  @test_util.run_v1_only("b/124229375")
  def testLoopStateWithTensorArray(self):
    with self.session(graph=ops.Graph()):
      max_time = 4
      batch_size = 16
      input_depth = 4
      num_units = 3

      inputs = np.random.randn(max_time, batch_size, input_depth)
      inputs_ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, size=array_ops.shape(inputs)[0])
      inputs_ta = inputs_ta.unstack(inputs)

      cell = rnn_cell.LSTMCell(num_units, state_is_tuple=True)

      def loop_fn(time_, cell_output, cell_state, loop_state):
        if cell_output is None:
          loop_state = tensor_array_ops.TensorArray(
              dynamic_size=True,
              size=0,
              dtype=dtypes.int32,
              clear_after_read=False)
          loop_state = loop_state.write(0, 1)
          next_state = cell.zero_state(batch_size, dtypes.float32)
        else:
          loop_state = loop_state.write(time_,
                                        loop_state.read(time_ - 1) + time_)
          next_state = cell_state
        emit_output = cell_output  # == None for time == 0
        elements_finished = array_ops.tile([time_ >= max_time], [batch_size])
        finished = math_ops.reduce_all(elements_finished)
        # For the very final iteration, we must emit a dummy input
        next_input = control_flow_ops.cond(
            finished,
            lambda: array_ops.zeros([batch_size, input_depth], dtype=dtypes.float32),
            lambda: inputs_ta.read(time_))
        return (elements_finished, next_input, next_state, emit_output,
                loop_state)

      r = rnn.raw_rnn(cell, loop_fn)
      loop_state = r[-1]
      loop_state = loop_state.stack()
      self.assertAllEqual([1, 2, 2 + 2, 4 + 3, 7 + 4], loop_state)

  @test_util.run_v1_only("b/124229375")
  def testEmitDifferentStructureThanCellOutput(self):
    with self.session(graph=ops.Graph()) as sess:
      max_time = 10
      batch_size = 16
      input_depth = 4
      num_units = 3

      inputs = np.random.randn(max_time, batch_size, input_depth)
      inputs_ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, size=array_ops.shape(inputs)[0])
      inputs_ta = inputs_ta.unstack(inputs)
      # Verify emit shapes may be unknown by feeding a placeholder that
      # determines an emit shape.
      unknown_dim = array_ops.placeholder(dtype=dtypes.int32)

      cell = rnn_cell.LSTMCell(num_units, state_is_tuple=True)

      def loop_fn(time_, cell_output, cell_state, _):
        if cell_output is None:
          emit_output = (array_ops.zeros([2, 3], dtype=dtypes.int32),
                         array_ops.zeros([unknown_dim], dtype=dtypes.int64))
          next_state = cell.zero_state(batch_size, dtypes.float32)
        else:
          emit_output = (array_ops.ones([batch_size, 2, 3], dtype=dtypes.int32),
                         array_ops.ones(
                             [batch_size, unknown_dim], dtype=dtypes.int64))
          next_state = cell_state
        elements_finished = array_ops.tile([time_ >= max_time], [batch_size])
        finished = math_ops.reduce_all(elements_finished)
        # For the very final iteration, we must emit a dummy input
        next_input = control_flow_ops.cond(
            finished,
            lambda: array_ops.zeros([batch_size, input_depth], dtype=dtypes.float32),
            lambda: inputs_ta.read(time_))
        return (elements_finished, next_input, next_state, emit_output, None)

      r = rnn.raw_rnn(cell, loop_fn)
      output_ta = r[0]
      self.assertEqual(2, len(output_ta))
      self.assertEqual([dtypes.int32, dtypes.int64],
                       [ta.dtype for ta in output_ta])
      output = [ta.stack() for ta in output_ta]
      output_vals = sess.run(output, feed_dict={unknown_dim: 1})
      self.assertAllEqual(
          np.ones((max_time, batch_size, 2, 3), np.int32), output_vals[0])
      self.assertAllEqual(
          np.ones((max_time, batch_size, 1), np.int64), output_vals[1])

  def _testScope(self, factory, prefix="prefix", use_outer_scope=True):
    with self.session(graph=ops.Graph()):
      if use_outer_scope:
        with variable_scope.variable_scope(prefix) as scope:
          factory(scope)
      else:
        factory(prefix)
        variables_lib.global_variables_initializer()

      # check that all the variables names starts
      # with the proper scope.
      all_vars = variables_lib.global_variables()
      prefix = prefix or "rnn"
      scope_vars = [v for v in all_vars if v.name.startswith(prefix + "/")]
      tf_logging.info("RNN with scope: %s (%s)" %
                      (prefix, "scope" if use_outer_scope else "str"))
      for v in scope_vars:
        tf_logging.info(v.name)
      self.assertEqual(len(scope_vars), len(all_vars))

  @test_util.run_v1_only("b/124229375")
  def testRawRNNScope(self):
    max_time = 10
    batch_size = 16
    input_depth = 4
    num_units = 3

    def factory(scope):
      inputs = array_ops.placeholder(
          shape=(max_time, batch_size, input_depth), dtype=dtypes.float32)
      sequence_length = array_ops.placeholder(
          shape=(batch_size,), dtype=dtypes.int32)
      inputs_ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, size=array_ops.shape(inputs)[0])
      inputs_ta = inputs_ta.unstack(inputs)

      cell = rnn_cell.LSTMCell(num_units, state_is_tuple=True)

      def loop_fn(time_, cell_output, cell_state, unused_loop_state):
        emit_output = cell_output  # == None for time == 0
        if cell_output is None:  # time == 0
          next_state = cell.zero_state(batch_size, dtypes.float32)
        else:
          next_state = cell_state

        elements_finished = (time_ >= sequence_length)
        finished = math_ops.reduce_all(elements_finished)
        # For the very final iteration, we must emit a dummy input
        next_input = control_flow_ops.cond(
            finished,
            lambda: array_ops.zeros([batch_size, input_depth], dtype=dtypes.float32),
            lambda: inputs_ta.read(time_))
        return (elements_finished, next_input, next_state, emit_output, None)

      return rnn.raw_rnn(cell, loop_fn, scope=scope)

    self._testScope(factory, use_outer_scope=True)
    self._testScope(factory, use_outer_scope=False)
    self._testScope(factory, prefix=None, use_outer_scope=False)


class DeviceWrapperCell(rnn_cell.RNNCell):
  """Class to ensure cell calculation happens on a specific device."""

  def __init__(self, cell, device):
    self._cell = cell
    self._device = device

  @property
  def output_size(self):
    return self._cell.output_size

  @property
  def state_size(self):
    return self._cell.state_size

  def __call__(self, input_, state, scope=None):
    if self._device is not None:
      with ops.device(self._device):
        return self._cell(input_, state, scope=scope)
    else:
      return self._cell(input_, state, scope=scope)


class TensorArrayOnCorrectDeviceTest(test.TestCase):

  def _execute_rnn_on(self,
                      rnn_device=None,
                      cell_device=None,
                      input_device=None):
    batch_size = 3
    time_steps = 7
    input_size = 5
    num_units = 10

    cell = rnn_cell.LSTMCell(num_units, use_peepholes=True)
    gpu_cell = DeviceWrapperCell(cell, cell_device)
    inputs = np.random.randn(batch_size, time_steps, input_size).astype(
        np.float32)
    sequence_length = np.random.randint(0, time_steps, size=batch_size)

    if input_device is not None:
      with ops.device(input_device):
        inputs = constant_op.constant(inputs)

    if rnn_device is not None:
      with ops.device(rnn_device):
        outputs, _ = rnn.dynamic_rnn(
            gpu_cell,
            inputs,
            sequence_length=sequence_length,
            dtype=dtypes.float32)
    else:
      outputs, _ = rnn.dynamic_rnn(
          gpu_cell,
          inputs,
          sequence_length=sequence_length,
          dtype=dtypes.float32)

    with self.session() as sess:
      opts = config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
      run_metadata = config_pb2.RunMetadata()
      variables_lib.global_variables_initializer().run()
      sess.run(outputs, options=opts, run_metadata=run_metadata)

    return run_metadata

  def _retrieve_cpu_gpu_stats(self, run_metadata):
    cpu_stats = None
    gpu_stats = None
    step_stats = run_metadata.step_stats
    for ds in step_stats.dev_stats:
      if "cpu:0" in ds.device[-5:].lower():
        cpu_stats = ds.node_stats
      if "gpu:0" == ds.device[-5:].lower():
        gpu_stats = ds.node_stats
    return cpu_stats, gpu_stats

  @test_util.run_v1_only("b/124229375")
  def testRNNOnCPUCellOnGPU(self):
    if not test.is_gpu_available():
      return  # Test requires access to a GPU

    gpu_dev = test.gpu_device_name()
    run_metadata = self._execute_rnn_on(
        rnn_device="/cpu:0", cell_device=gpu_dev)
    cpu_stats, gpu_stats = self._retrieve_cpu_gpu_stats(run_metadata)

    def _assert_in(op_str, in_stats, out_stats):
      self.assertTrue(any(op_str in s.node_name for s in in_stats))
      self.assertFalse(any(op_str in s.node_name for s in out_stats))

    # Writes happen at output of RNN cell
    _assert_in("TensorArrayWrite", gpu_stats, cpu_stats)
    # Gather happens on final TensorArray
    _assert_in("TensorArrayGather", gpu_stats, cpu_stats)
    # Reads happen at input to RNN cell
    _assert_in("TensorArrayRead", cpu_stats, gpu_stats)
    # Scatters happen to get initial input into TensorArray
    _assert_in("TensorArrayScatter", cpu_stats, gpu_stats)

  @test_util.run_v1_only("b/124229375")
  def testRNNOnCPUCellOnCPU(self):
    if not test.is_gpu_available():
      return  # Test requires access to a GPU

    gpu_dev = test.gpu_device_name()
    run_metadata = self._execute_rnn_on(
        rnn_device="/cpu:0", cell_device="/cpu:0", input_device=gpu_dev)
    cpu_stats, gpu_stats = self._retrieve_cpu_gpu_stats(run_metadata)

    def _assert_in(op_str, in_stats, out_stats):
      self.assertTrue(any(op_str in s.node_name for s in in_stats))
      self.assertFalse(any(op_str in s.node_name for s in out_stats))

    # All TensorArray operations happen on CPU
    _assert_in("TensorArray", cpu_stats, gpu_stats)

  @test_util.run_v1_only("b/124229375")
  def testInputOnGPUCellNotDeclared(self):
    if not test.is_gpu_available():
      return  # Test requires access to a GPU

    gpu_dev = test.gpu_device_name()
    run_metadata = self._execute_rnn_on(input_device=gpu_dev)
    cpu_stats, gpu_stats = self._retrieve_cpu_gpu_stats(run_metadata)

    def _assert_in(op_str, in_stats, out_stats):
      self.assertTrue(any(op_str in s.node_name for s in in_stats))
      self.assertFalse(any(op_str in s.node_name for s in out_stats))

    # Everything happens on GPU
    _assert_in("TensorArray", gpu_stats, cpu_stats)


if __name__ == "__main__":
  test.main()