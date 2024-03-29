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
"""Functional test for moving_averages.py."""

import numpy as np

from tensorflow.python.compiler.xla.experimental import xla_sharding
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import moving_averages
from tensorflow.python.training import saver as saver_lib
from ..utils.timer_wrapper import tensorflow_op_timer



def _Repeat(value, dim):
  if dim == 1:
    return value
  return [value] * dim


class ExponentialMovingAverageTest(test.TestCase):

  def _CheckDecay(self, ema, actual_decay, dim, dynamic_decay_value=None):

    def _Scale(dk, steps):
      if ema._zero_debias:
        return 1 - dk**steps
      else:
        return 1

    tens = _Repeat(10.0, dim)
    thirties = _Repeat(30.0, dim)
    var0 = variables.Variable(tens, name="v0")
    var1 = variables.Variable(thirties, name="v1")
    self.evaluate(variables.global_variables_initializer())
    # Note that tensor2 is not a Variable but just a plain Tensor resulting
    # from the sum operation.
    tensor2 = var0 + var1
    if dynamic_decay_value is not None:
      self.evaluate(ema._decay.assign(dynamic_decay_value))
    update = ema.apply([var0, var1, tensor2])
    avg0 = ema.average(var0)
    avg1 = ema.average(var1)
    avg2 = ema.average(tensor2)

    self.assertItemsEqual([var0, var1], variables.moving_average_variables())

    self.assertNotIn(avg0, variables.trainable_variables())
    self.assertNotIn(avg1, variables.trainable_variables())
    self.assertNotIn(avg2, variables.trainable_variables())
    self.evaluate(variables.global_variables_initializer())
    if dynamic_decay_value is not None:
      self.evaluate(ema._decay.assign(dynamic_decay_value))

    self.assertEqual("v0/ExponentialMovingAverage:0", avg0.name)
    self.assertEqual("v1/ExponentialMovingAverage:0", avg1.name)
    self.assertEqual("add/ExponentialMovingAverage:0", avg2.name)

    # Check initial values.
    self.assertAllClose(tens, self.evaluate(var0))
    self.assertAllClose(thirties, self.evaluate(var1))
    self.assertAllClose(_Repeat(10.0 + 30.0, dim), self.evaluate(tensor2))

    # Check that averages are initialized correctly.
    self.assertAllClose(tens, self.evaluate(avg0))
    self.assertAllClose(thirties, self.evaluate(avg1))
    # Note that averages of Tensor's initialize to zeros_like since no value
    # of the Tensor is known because the Op has not been run (yet).
    self.assertAllClose(_Repeat(0.0, dim), self.evaluate(avg2))

    # Update the averages and check.
    self.evaluate(update)
    dk = actual_decay

    expected = _Repeat(10.0 * dk + 10.0 * (1 - dk), dim)
    self.assertAllClose(expected, self.evaluate(avg0))
    expected = _Repeat(30.0 * dk + 30.0 * (1 - dk), dim)
    self.assertAllClose(expected, self.evaluate(avg1))
    expected = _Repeat(0.0 * dk + (10.0 + 30.0) * (1 - dk) / _Scale(dk, 1), dim)
    self.assertAllClose(expected, self.evaluate(avg2))

    # Again, update the averages and check.
    self.evaluate(update)
    expected = _Repeat((10.0 * dk + 10.0 * (1 - dk)) * dk + 10.0 * (1 - dk),
                       dim)
    self.assertAllClose(expected, self.evaluate(avg0))
    expected = _Repeat((30.0 * dk + 30.0 * (1 - dk)) * dk + 30.0 * (1 - dk),
                       dim)
    self.assertAllClose(expected, self.evaluate(avg1))
    expected = _Repeat(((0.0 * dk + (10.0 + 30.0) * (1 - dk)) * dk +
                        (10.0 + 30.0) * (1 - dk)) / _Scale(dk, 2), dim)
    self.assertAllClose(expected, self.evaluate(avg2))

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNoNumUpdates_Scalar(self):
    timer = tensorflow_op_timer()
    with timer:
      ema = moving_averages.ExponentialMovingAverage(0.25)
      timer.gen.send(ema)
    self._CheckDecay(ema, actual_decay=0.25, dim=1)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNoNumUpdates_Scalar_DynamicDecay(self):
    decay_var = variables.Variable(0.75)
    timer = tensorflow_op_timer()
    with timer:
      ema = moving_averages.ExponentialMovingAverage(decay_var)
      timer.gen.send(ema)
    self._CheckDecay(ema, actual_decay=0.25, dim=1, dynamic_decay_value=0.25)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNoNumUpdates_Scalar_Debias(self):
    timer = tensorflow_op_timer()
    with timer:
      ema = moving_averages.ExponentialMovingAverage(0.25, zero_debias=True)
      timer.gen.send(ema)
    self._CheckDecay(ema, actual_decay=0.25, dim=1)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNoNumUpdates_Scalar_Debias_DynamicDecay(self):
    decay_var = variables.Variable(0.75)
    timer = tensorflow_op_timer()
    with timer:
      ema = moving_averages.ExponentialMovingAverage(decay_var, zero_debias=True)
      timer.gen.send(ema)
    self._CheckDecay(ema, actual_decay=0.25, dim=1, dynamic_decay_value=0.25)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNoNumUpdates_Vector(self):
    timer = tensorflow_op_timer()
    with timer:
      ema = moving_averages.ExponentialMovingAverage(0.25)
      timer.gen.send(ema)
    self._CheckDecay(ema, actual_decay=0.25, dim=5)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNoNumUpdates_Vector_DynamicDecay(self):
    decay_var = variables.Variable(0.75)
    timer = tensorflow_op_timer()
    with timer:
      ema = moving_averages.ExponentialMovingAverage(decay_var)
      timer.gen.send(ema)
    self._CheckDecay(ema, actual_decay=0.25, dim=5, dynamic_decay_value=0.25)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNoNumUpdates_Vector_Debias(self):
    timer = tensorflow_op_timer()
    with timer:
      ema = moving_averages.ExponentialMovingAverage(0.25, zero_debias=True)
      timer.gen.send(ema)
    self._CheckDecay(ema, actual_decay=0.25, dim=5)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNoNumUpdates_Vector_Debias_DynamicDecay(self):
    decay_var = variables.Variable(0.75)
    timer = tensorflow_op_timer()
    with timer:
      ema = moving_averages.ExponentialMovingAverage(decay_var, zero_debias=True)
      timer.gen.send(ema)
    self._CheckDecay(ema, actual_decay=0.25, dim=5, dynamic_decay_value=0.25)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNumUpdates_Scalar(self):
    # With num_updates 1, the decay applied is 0.181818.
    timer = tensorflow_op_timer()
    with timer:
      ema = moving_averages.ExponentialMovingAverage(0.25, num_updates=1)
      timer.gen.send(ema)
    self._CheckDecay(ema, actual_decay=0.181818, dim=1)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNumUpdates_Scalar_DynamicDecay(self):
    decay_var = variables.Variable(0.75)
    # With num_updates 1, the decay applied is 0.181818.
    timer = tensorflow_op_timer()
    with timer:
      ema = moving_averages.ExponentialMovingAverage(decay_var, num_updates=1)
      timer.gen.send(ema)
    self._CheckDecay(
        ema, actual_decay=0.181818, dim=1, dynamic_decay_value=0.25)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNumUpdates_Scalar_Debias(self):
    # With num_updates 1, the decay applied is 0.181818.
    timer = tensorflow_op_timer()
    with timer:
      ema = moving_averages.ExponentialMovingAverage(
        0.25, num_updates=1, zero_debias=True)
      timer.gen.send(ema)
    self._CheckDecay(ema, actual_decay=0.181818, dim=1)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNumUpdates_Scalar_Debias_DynamicDecay(self):
    decay_var = variables.Variable(0.75)
    # With num_updates 1, the decay applied is 0.181818.
    timer = tensorflow_op_timer()
    with timer:
      ema = moving_averages.ExponentialMovingAverage(
        decay_var, num_updates=1, zero_debias=True)
      timer.gen.send(ema)
    self._CheckDecay(
        ema, actual_decay=0.181818, dim=1, dynamic_decay_value=0.25)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNumUpdates_Vector(self):
    # With num_updates 1, the decay applied is 0.181818.
    timer = tensorflow_op_timer()
    with timer:
      ema = moving_averages.ExponentialMovingAverage(0.25, num_updates=1)
      timer.gen.send(ema)
    self._CheckDecay(ema, actual_decay=0.181818, dim=5)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNumUpdates_Vector_DynamicDecay(self):
    decay_var = variables.Variable(0.75)
    # With num_updates 1, the decay applied is 0.181818.
    timer = tensorflow_op_timer()
    with timer:
      ema = moving_averages.ExponentialMovingAverage(decay_var, num_updates=1)
      timer.gen.send(ema)
    self._CheckDecay(
        ema, actual_decay=0.181818, dim=5, dynamic_decay_value=0.25)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNumUpdates_Vector_Debias(self):
    # With num_updates 1, the decay applied is 0.181818.
    timer = tensorflow_op_timer()
    with timer:
      ema = moving_averages.ExponentialMovingAverage(
        0.25, num_updates=1, zero_debias=True)
      timer.gen.send(ema)
    self._CheckDecay(ema, actual_decay=0.181818, dim=5)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNumUpdates_Vector_Debias_DynamicDecay(self):
    decay_var = variables.Variable(0.75)
    # With num_updates 1, the decay applied is 0.181818.
    timer = tensorflow_op_timer()
    with timer:
      ema = moving_averages.ExponentialMovingAverage(
        decay_var, num_updates=1, zero_debias=True)
      timer.gen.send(ema)
    self._CheckDecay(
        ema, actual_decay=0.181818, dim=5, dynamic_decay_value=0.25)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesWithControlDeps(self):
    v0 = variables.Variable(0, name="v0")
    add_to_v0 = v0.assign_add(1)
    v1 = variables.Variable([10.0], name="v1")
    assign_to_v1 = v1.assign([20.0])
    timer = tensorflow_op_timer()
    with timer:
      ema = moving_averages.ExponentialMovingAverage(0.25)
      timer.gen.send(ema)
    with ops.control_dependencies([add_to_v0]):
      ema_op = ema.apply([v1])
    # the moving average of v1 should not have any control inputs
    v1_avg = ema.average(v1)
    self.assertEqual([], v1_avg.initializer.control_inputs)
    self.assertEqual([], v1_avg.value().op.control_inputs)
    self.assertEqual([], v1_avg.value().op.control_inputs)
    # We should be able to initialize v1_avg before v0.
    self.evaluate(v1_avg.initializer)
    self.evaluate(v0.initializer)
    self.assertEqual([10.0], self.evaluate(v1_avg))
    # running ema_op should add to v0 (in addition to updating v1_avg)
    self.evaluate(assign_to_v1)
    self.evaluate(ema_op)
    self.assertEqual(1, self.evaluate(v0))
    self.assertEqual([17.5], self.evaluate(v1_avg))

  def testBasicEager(self):
    v0 = variables.Variable(1.0, name="v0")
    v1 = variables.Variable(2.0, name="v1")
    timer = tensorflow_op_timer()
    with timer:  
      ema = moving_averages.ExponentialMovingAverage(0.25, name="foo")
      timer.gen.send(ema)
    op = ema.apply([v0, v1])
    if not context.executing_eagerly():
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(op)

    self.evaluate(v0.assign(2.0))
    self.evaluate(v1.assign(4.0))

    self.evaluate(ema.apply([v0, v1]))

    self.assertEqual("foo", ema.name)
    self.assertEqual("v0/foo", ema.average_name(v0))
    self.assertEqual("v1/foo", ema.average_name(v1))

    self.assertAllEqual(self.evaluate(ema.average(v0)), 1.75)
    self.assertAllEqual(self.evaluate(ema.average(v1)), 3.5)

  def averageVariablesNamesHelper(self, zero_debias):
    v0 = variables.Variable(10.0, name="v0")
    v1 = variables.Variable(30.0, name="v1")
    # Add a non-trainable variable.
    v2 = variables.Variable(20.0, name="v2", trainable=False)
    tensor2 = v0 + v1
    timer = tensorflow_op_timer()
    with timer:
      ema = moving_averages.ExponentialMovingAverage(
        0.25, zero_debias=zero_debias, name="foo")
      timer.gen.send(ema)
    self.assertEqual("foo", ema.name)
    self.assertEqual("v0/foo", ema.average_name(v0))
    self.assertEqual("v1/foo", ema.average_name(v1))
    self.assertEqual("add/foo", ema.average_name(tensor2))
    ema.apply([v0, v1, tensor2])
    vars_to_restore = ema.variables_to_restore()
    # vars_to_restore should contain the following:
    # {v0/foo : v0,
    #  v1/foo : v1,
    #  add/foo : add/foo,
    #  v2 : v2}
    expected_names = [
        ema.average_name(v0),
        ema.average_name(v1),
        ema.average_name(tensor2), v2.op.name
    ]
    if zero_debias:
      # vars_to_restore should also contain the following:
      #  {add/foo/biased: add/foo/biased,
      #  add/foo/local_step: add/foo/local_step}
      expected_names += [
          ema.average_name(tensor2) + "/biased",
          ema.average_name(tensor2) + "/local_step"
      ]
    self.assertEqual(sorted(expected_names), sorted(vars_to_restore.keys()))
    self.assertEqual(ema.average(v0).op.name, ema.average_name(v0))
    self.assertEqual(ema.average(v1).op.name, ema.average_name(v1))
    self.assertEqual(ema.average(tensor2).op.name, ema.average_name(tensor2))

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNames(self):
    self.averageVariablesNamesHelper(zero_debias=True)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNamesNoDebias(self):
    self.averageVariablesNamesHelper(zero_debias=False)

  @test_util.deprecated_graph_mode_only
  def averageVariablesNamesRespectScopeHelper(self, zero_debias):
    # See discussion on #2740.
    with variable_scope.variable_scope("scope1"):
      v0 = variables.Variable(10.0, name="v0")
      v1 = variables.Variable(30.0, name="v1")
      # Add a non-trainable variable.
      v2 = variables.Variable(20.0, name="v2", trainable=False)
      tensor2 = v0 + v1
    with variable_scope.variable_scope("scope2"):
      timer = tensorflow_op_timer()
      with timer:
        ema = moving_averages.ExponentialMovingAverage(
          0.25, zero_debias=zero_debias, name="foo")
        timer.gen.send(ema)
      self.assertEqual("scope2/scope1/v0/foo", ema.average_name(v0))
      self.assertEqual("scope2/scope1/v1/foo", ema.average_name(v1))
      self.assertEqual("scope2/scope1/add/foo", ema.average_name(tensor2))
      ema.apply([v0, v1, tensor2])
      vars_to_restore = ema.variables_to_restore()
      # `vars_to_restore` should contain the following:
      # {scope2/scope1/v0/foo : v0,
      #  scope2/scope1/v1/foo : v1,
      #  scope2/scope1/add/foo : add/foo,
      #  scope1/v2 : v2}
      expected_names = [
          ema.average_name(v0),
          ema.average_name(v1),
          ema.average_name(tensor2), v2.op.name
      ]
      if zero_debias:
        # `vars_to_restore` should also contain the following:
        # {scope2/scope2/scope1/add/foo/biased: add/foo/biased,
        #  scope2/scope2/scope1/add/foo/local_step: add/foo/local_step}
        sc = "scope2/"
        expected_names += [
            sc + ema.average_name(tensor2) + "/biased",
            sc + ema.average_name(tensor2) + "/local_step"
        ]

      self.assertEqual(sorted(expected_names), sorted(vars_to_restore.keys()))
      self.assertEqual(ema.average(v0).op.name, ema.average_name(v0))
      self.assertEqual(ema.average(v1).op.name, ema.average_name(v1))
      self.assertEqual(ema.average(tensor2).op.name, ema.average_name(tensor2))

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNamesRespectScope(self):
    self.averageVariablesNamesRespectScopeHelper(zero_debias=True)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNamesRespectScopeNoDebias(self):
    self.averageVariablesNamesRespectScopeHelper(zero_debias=False)

  @test_util.deprecated_graph_mode_only
  def testSubsetAverageVariablesNames(self):
    v0 = variables.Variable(10.0, name="v0")
    v1 = variables.Variable(30.0, name="v1")
    # Add a non-trainable variable.
    v2 = variables.Variable(20.0, name="v2", trainable=False)
    tensor2 = v0 + v1
    timer = tensorflow_op_timer()
    with timer:
      ema = moving_averages.ExponentialMovingAverage(0.25, name="foo_avg")
      timer.gen.send(ema)
    self.assertEqual("v0/foo_avg", ema.average_name(v0))
    self.assertEqual("v1/foo_avg", ema.average_name(v1))
    self.assertEqual("add/foo_avg", ema.average_name(tensor2))
    vars_to_restore = ema.variables_to_restore([v0, tensor2])
    # vars_to_restore should contain the following:
    # {v0/foo_avg : v0,
    #  add/foo_avg : add
    #  v1 : v1,
    #  v2 : v2}
    self.assertEqual(
        sorted(vars_to_restore.keys()),
        sorted([
            ema.average_name(v0),
            ema.average_name(tensor2), v1.op.name, v2.op.name
        ]))
    ema.apply([v0, v1, tensor2])
    self.assertEqual(ema.average(v0).op.name, ema.average_name(v0))
    self.assertEqual(ema.average(v1).op.name, ema.average_name(v1))
    self.assertEqual(ema.average(tensor2).op.name, ema.average_name(tensor2))

  def testSubsetAverageVariablesNamesEager(self):
    v0 = variables.Variable(10.0, name="v0")
    v1 = variables.Variable(30.0, name="v1")
    # Add a non-trainable variable.
    v2 = variables.Variable(20.0, name="v2", trainable=False)
    timer = tensorflow_op_timer()
    with timer:
      ema = moving_averages.ExponentialMovingAverage(0.25, name="foo_avg")
      timer.gen.send(ema)
    self.assertEqual("v0/foo_avg", ema.average_name(v0))
    self.assertEqual("v1/foo_avg", ema.average_name(v1))
    vars_to_restore = ema.variables_to_restore([v0, v1, v2])
    self.assertAllEqual(
        sorted(vars_to_restore.keys()),
        sorted([
            ema.average_name(v0), ema.average_name(v1), ema.average_name(v2)
        ]))
    ema.apply([v0, v1])
    self.assertEqual(ema.average(v0).name[:-len(":0")], ema.average_name(v0))
    self.assertEqual(ema.average(v1).name[:-len(":0")], ema.average_name(v1))

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesDeviceAssignment(self):
    with ops.device("/job:dev_v0"):
      v0 = variables.Variable(10.0, name="v0")
    with ops.device("/job:dev_v1"):
      v1 = gen_state_ops.variable(
          shape=[1],
          dtype=dtypes.float32,
          name="v1",
          container="",
          shared_name="")
      v1.set_shape([1])
    tensor2 = v0 + v1
    timer = tensorflow_op_timer()
    with timer:
      ema = moving_averages.ExponentialMovingAverage(0.25, name="foo_avg")
      timer.gen.send(ema)
    with ops.device("/job:default"):
      ema.apply([v0, v1, tensor2])
    self.assertDeviceEqual("/job:dev_v0", ema.average(v0).device)
    self.assertDeviceEqual("/job:dev_v1", ema.average(v1).device)
    # However, the colocation property is maintained.
    self.assertEqual([b"loc:@v1"], ema.average(v1).op.colocation_groups())
    self.assertDeviceEqual("/job:default", ema.average(tensor2).device)

  def _ExportAndImportGraph(self, graph):
    """Export and import graph into a new graph."""
    meta_graph = saver_lib.export_meta_graph(
        graph=graph, collection_list=graph.get_all_collection_keys())
    graph_copy = ops.Graph()
    with graph_copy.as_default():
      _ = saver_lib.import_meta_graph(meta_graph)
    return graph_copy

  @test_util.deprecated_graph_mode_only
  def testImportedGraphVariablesToRestore(self):
    g = ops.Graph()
    with g.as_default():
      variables.Variable(10.0, name="v")
    # Export and import the graph into a new graph.
    g_copy = self._ExportAndImportGraph(g)
    with g_copy.as_default():
      timer = tensorflow_op_timer()
      with timer:
        ema = moving_averages.ExponentialMovingAverage(0.25, name="foo_avg")
        timer.gen.send(ema)
      vars_to_restore = ema.variables_to_restore()
      # There should only be one variable in vars_to_restore. This is important
      # to check because when importing from a GraphDef, TF makes duplicate
      # python Variable objects referring to the same underlying variable. We
      # need to be sure that two variables referring to the same variable don't
      # both get added to vars_to_restore.
      self.assertEqual(len(vars_to_restore), 1)
      self.assertIn("v/foo_avg", vars_to_restore)

  @test_util.deprecated_graph_mode_only
  def testCopyXlaSharding(self):
    timer = tensorflow_op_timer()
    with timer:
      ema = moving_averages.ExponentialMovingAverage(0.25, name="foo_avg")
      timer.gen.send(ema)
    v = variables.Variable(_Repeat(10.0, 2), name="v")
    self.assertIsNone(xla_sharding.get_tensor_sharding(v))
    v = xla_sharding.mesh_split(v, np.array([0, 1]), [0], use_sharding_op=False)
    self.assertIsNotNone(xla_sharding.get_tensor_sharding(v))
    self.evaluate(variables.global_variables_initializer())
    ema.apply([v])
    avg = ema.average(v)
    self.assertEqual(
        xla_sharding.get_tensor_sharding(v),
        xla_sharding.get_tensor_sharding(avg))


if __name__ == "__main__":
  test.main()
