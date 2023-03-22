"""Tests for tf.py."""

import functools
import operator

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent
from tensorflow.python.util import compat


class VariablesTestCase(test.TestCase, parameterized.TestCase):

  @test_util.run_v1_only("b/120545219")
  def testInitialization(self):
    with self.cached_session():
      var0 = variables.VariableV1(0.0)
      self.assertEqual("Variable:0", var0.name)
      self.assertEqual("Variable", var0._shared_name)
      self.assertEqual([], var0.get_shape())
      self.assertEqual([], var0.get_shape())
      self.assertEqual([], var0.shape)

      var1 = variables.VariableV1(1.1)
      self.assertEqual("Variable_1:0", var1.name)
      self.assertEqual("Variable_1", var1._shared_name)
      self.assertEqual([], var1.get_shape())
      self.assertEqual([], var1.get_shape())
      self.assertEqual([], var1.shape)

      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        self.evaluate(var0)

      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        self.evaluate(var1)

      self.evaluate(variables.global_variables_initializer())

      self.assertAllClose(0.0, self.evaluate(var0))
      self.assertAllClose(1.1, self.evaluate(var1))

  @test_util.run_v1_only("b/120545219")
  def testInitializationOrder(self):
    with self.cached_session():
      rnd = variables.Variable(random_ops.random_uniform([3, 6]), name="rnd")
      self.assertEqual("rnd:0", rnd.name)
      self.assertEqual([3, 6], rnd.get_shape())
      self.assertEqual([3, 6], rnd.get_shape())
      self.assertEqual([3, 6], rnd.shape)

      dep = variables.Variable(rnd.initialized_value(), name="dep")
      self.assertEqual("dep:0", dep.name)
      self.assertEqual([3, 6], dep.get_shape())
      self.assertEqual([3, 6], dep.get_shape())
      self.assertEqual([3, 6], dep.shape)

      # Currently have to set the shape manually for Add.
      added_val = rnd.initialized_value() + dep.initialized_value() + 2.0
      added_val.set_shape(rnd.get_shape())

      depdep = variables.Variable(added_val, name="depdep")
      self.assertEqual("depdep:0", depdep.name)
      self.assertEqual([3, 6], depdep.get_shape())
      self.assertEqual([3, 6], depdep.get_shape())
      self.assertEqual([3, 6], depdep.shape)

      self.evaluate(variables.global_variables_initializer())

      self.assertAllClose(self.evaluate(rnd), self.evaluate(dep))
      self.assertAllClose(
          self.evaluate(rnd) + self.evaluate(dep) + 2.0, self.evaluate(depdep))


  @test_util.run_deprecated_v1
  def testAssignments(self):
    with self.cached_session():
      var = variables.Variable(0.0)
      plus_one = var.assign_add(1.0)
      minus_one = var.assign_sub(2.0)
      four = var.assign(4.0)
      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose(0.0, self.evaluate(var))

      self.assertAllClose(1.0, self.evaluate(plus_one))
      self.assertAllClose(1.0, self.evaluate(var))

      self.assertAllClose(-1.0, self.evaluate(minus_one))
      self.assertAllClose(-1.0, self.evaluate(var))

      self.assertAllClose(4.0, self.evaluate(four))
      self.assertAllClose(4.0, self.evaluate(var))

  @test_util.run_deprecated_v1
  def testResourceAssignments(self):
    with self.session():
      var = resource_variable_ops.ResourceVariable(0.0)
      plus_one = var.assign_add(1.0)
      minus_one = var.assign_sub(2.0)
      four = var.assign(4.0)
      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose(0.0, self.evaluate(var))

      self.evaluate(plus_one)
      self.assertAllClose(1.0, self.evaluate(var))

      self.evaluate(minus_one)
      self.assertAllClose(-1.0, self.evaluate(var))

      self.evaluate(four)
      self.assertAllClose(4.0, self.evaluate(var))

  @test_util.run_in_graph_and_eager_modes
  def testAssignDifferentShapesAllowed(self):
    var = variables.Variable(np.zeros(shape=[1, 1]),
                             shape=tensor_shape.TensorShape(None))
    print("***VAR", var.device)
    self.evaluate(variables.global_variables_initializer())
    self.assertAllEqual(np.zeros(shape=[1, 1]), var.read_value())
    self.evaluate(var.assign(np.zeros(shape=[2, 2])))
    self.assertAllEqual(np.zeros(shape=[2, 2]), var.read_value())


  def _countUpToTest(self, dtype):
    with self.cached_session():
      zero = constant_op.constant(0, dtype=dtype)
      var = variables.Variable(zero)
      count_up_to = var.count_up_to(3)

      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(0, self.evaluate(var))

      self.assertEqual(0, self.evaluate(count_up_to))
      self.assertEqual(1, self.evaluate(var))

      self.assertEqual(1, self.evaluate(count_up_to))
      self.assertEqual(2, self.evaluate(var))

      self.assertEqual(2, self.evaluate(count_up_to))
      self.assertEqual(3, self.evaluate(var))

      with self.assertRaisesOpError("Reached limit of 3"):
        self.evaluate(count_up_to)
      self.assertEqual(3, self.evaluate(var))

      with self.assertRaisesOpError("Reached limit of 3"):
        self.evaluate(count_up_to)
      self.assertEqual(3, self.evaluate(var))

  @test_util.run_deprecated_v1
  def testUseVariableAsTensor(self):
    with self.cached_session():
      var_x = variables.Variable(2.0)
      var_y = variables.Variable(3.0)
      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose(2.0, self.evaluate(var_x))
      self.assertAllClose(3.0, self.evaluate(var_y))
      self.assertAllClose(5.0, self.evaluate(math_ops.add(var_x, var_y)))

  @test_util.run_deprecated_v1
  def testZeroSizeVarSameAsConst(self):
    with self.cached_session():
      zero_size_var = variables.Variable(array_ops.zeros([0, 2]))
      zero_size_const = array_ops.ones([2, 0])
      variable_mul = math_ops.matmul(zero_size_const, zero_size_var)
      const_mul = math_ops.matmul(
          zero_size_const, zero_size_const, transpose_b=True)
      self.evaluate(variables.global_variables_initializer())
      variable_output = self.evaluate(variable_mul)
      self.assertAllClose(self.evaluate(const_mul), variable_output)
      self.assertAllClose([[0., 0.], [0., 0.]], variable_output)

  @test_util.run_deprecated_v1
  def testOperators(self):
    with self.cached_session():
      var_f = variables.Variable([2.0])
      add = var_f + 0.0
      radd = 1.0 + var_f
      sub = var_f - 1.0
      rsub = 1.0 - var_f
      mul = var_f * 10.0
      rmul = 10.0 * var_f
      div = var_f / 10.0
      rdiv = 10.0 / var_f
      lt = var_f < 3.0
      rlt = 3.0 < var_f
      le = var_f <= 2.0
      rle = 2.0 <= var_f
      gt = var_f > 3.0
      rgt = 3.0 > var_f
      ge = var_f >= 2.0
      rge = 2.0 >= var_f
      neg = -var_f
      abs_v = abs(var_f)

      var_i = variables.Variable([20])
      mod = var_i % 7
      rmod = 103 % var_i

      var_b = variables.Variable([True, False])
      and_v = operator.and_(var_b, [True, True])
      or_v = operator.or_(var_b, [False, True])
      xor_v = operator.xor(var_b, [False, False])
      invert_v = ~var_b

      rnd = np.random.rand(4, 4).astype("f")
      var_t = variables.Variable(rnd)
      slice_v = var_t[2, 0:0]

      var_m = variables.Variable([[2.0, 3.0]])
      matmul = var_m.__matmul__([[10.0], [20.0]])
      rmatmul = var_m.__rmatmul__([[10.0], [20.0]])

      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose([2.0], self.evaluate(add))
      self.assertAllClose([3.0], self.evaluate(radd))
      self.assertAllClose([1.0], self.evaluate(sub))
      self.assertAllClose([-1.0], self.evaluate(rsub))
      self.assertAllClose([20.0], self.evaluate(mul))
      self.assertAllClose([20.0], self.evaluate(rmul))
      self.assertAllClose([0.2], self.evaluate(div))
      self.assertAllClose([5.0], self.evaluate(rdiv))
      self.assertAllClose([-2.0], self.evaluate(neg))
      self.assertAllClose([2.0], self.evaluate(abs_v))
      self.assertAllClose([True], self.evaluate(lt))
      self.assertAllClose([False], self.evaluate(rlt))
      self.assertAllClose([True], self.evaluate(le))
      self.assertAllClose([True], self.evaluate(rle))
      self.assertAllClose([False], self.evaluate(gt))
      self.assertAllClose([True], self.evaluate(rgt))
      self.assertAllClose([True], self.evaluate(ge))
      self.assertAllClose([True], self.evaluate(rge))

      self.assertAllClose([6], self.evaluate(mod))
      self.assertAllClose([3], self.evaluate(rmod))

      self.assertAllClose([True, False], self.evaluate(and_v))
      self.assertAllClose([True, True], self.evaluate(or_v))
      self.assertAllClose([True, False], self.evaluate(xor_v))
      self.assertAllClose([False, True], self.evaluate(invert_v))

      self.assertAllClose(rnd[2, 0:0], self.evaluate(slice_v))

      self.assertAllClose([[80.0]], self.evaluate(matmul))
      self.assertAllClose([[20.0, 30.0], [40.0, 60.0]], self.evaluate(rmatmul))

  @test_util.run_deprecated_v1
  def testSession(self):
    with self.cached_session() as sess:
      var = variables.Variable([1, 12])
      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose([1, 12], self.evaluate(var))

  @test_util.run_v1_only("b/120545219")
  def testInitializerFunction(self):
    value = [[-42], [133.7]]
    shape = [2, 1]
    with self.cached_session():
      initializer = lambda: constant_op.constant(value)

      v1 = variables.Variable(initializer, dtype=dtypes.float32)
      self.assertEqual(shape, v1.get_shape())
      self.assertEqual(shape, v1.shape)
      self.assertAllClose(value, self.evaluate(v1.initial_value))
      with self.assertRaises(errors_impl.FailedPreconditionError):
        self.evaluate(v1)

      v2 = variables.Variable(
          math_ops.negative(v1.initialized_value()), dtype=dtypes.float32)
      self.assertEqual(v1.get_shape(), v2.get_shape())
      self.assertEqual(v1.shape, v2.shape)
      self.assertAllClose(np.negative(value), self.evaluate(v2.initial_value))

      with self.assertRaises(errors_impl.FailedPreconditionError):
        self.evaluate(v2)
      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose(np.negative(value), self.evaluate(v2))

  @test_util.run_v1_only("b/120545219")
  def testNoRefDataRace(self):
    with self.cached_session():
      a = variables.Variable([1, 2, 3], dtype=dtypes.float32)
      b = variables.Variable(a.initialized_value() + 2)
      c = variables.Variable(b.initialized_value() + 2)
      self.evaluate(variables.global_variables_initializer())
      self.assertAllEqual(self.evaluate(a), [1, 2, 3])
      self.assertAllEqual(self.evaluate(b), [3, 4, 5])
      self.assertAllEqual(self.evaluate(c), [5, 6, 7])

  @test_util.run_deprecated_v1
  def testLoad(self):
    with self.cached_session():
      var = variables.Variable(np.zeros((5, 5), np.float32))
      self.evaluate(variables.global_variables_initializer())
      var.load(np.ones((5, 5), np.float32))

      self.assertAllClose(np.ones((5, 5), np.float32), self.evaluate(var))


class IsInitializedTest(test.TestCase):

  def testAssertVariablesInitialized(self):
    with ops.Graph().as_default(), self.cached_session() as sess:
      v = variables.Variable([1, 2], name="v")
      w = variables.Variable([3, 4], name="w")
      _ = v, w
      uninited = variables.report_uninitialized_variables()
      self.assertAllEqual(np.array([b"v", b"w"]), self.evaluate(uninited))
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(0, self.evaluate(uninited).size)


  def testTrainingWithZeroSizeVar(self):
    with ops.Graph().as_default(), self.cached_session() as sess:
      a = variables.Variable(array_ops.zeros([0, 2]))
      b = variables.Variable(array_ops.ones([2, 2]))
      print("***A", a.device, sess.graph.device)
      objective = math_ops.reduce_sum(b + math_ops.matmul(
          a, a, transpose_a=True))
      print("INITIALIZER", variables.global_variables_initializer())
      self.evaluate(variables.global_variables_initializer())
      do_opt = gradient_descent.GradientDescentOptimizer(0.1).minimize(
          objective)
      self.evaluate([do_opt])
      self.assertAllClose([[0.9, 0.9], [0.9, 0.9]], self.evaluate(b))


@test_util.run_v1_only("b/120545219")
class ObsoleteIsInitializedTest(test.TestCase):

  def testVariables(self):
    with ops.Graph().as_default(), self.cached_session() as sess:
      v = variables.VariableV1([1, 2])
      w = variables.VariableV1([3, 4])
      _ = v, w
      inited = variables.assert_variables_initialized()
      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        self.evaluate(inited)
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(inited)

  def testPartitionedVariableAssignments(self):
    with ops.Graph().as_default(), self.cached_session():
      v0 = variables.Variable(initial_value=[0.0])
      v1 = variables.Variable(initial_value=[1.0])
      v2 = variables.Variable(initial_value=[20.0])
      v3 = variables.Variable(initial_value=[30.0])
      v0._set_save_slice_info(
          variables.Variable.SaveSliceInfo(v0.name, [2], [0], [1]))
      v1._set_save_slice_info(
          variables.Variable.SaveSliceInfo(v1.name, [2], [1], [1]))
      v2._set_save_slice_info(
          variables.Variable.SaveSliceInfo(v2.name, [2], [0], [1]))
      v3._set_save_slice_info(
          variables.Variable.SaveSliceInfo(v3.name, [2], [1], [1]))

      partitions = [2]

      # Pass variable_list as [v1, v0] to ensure they are properly
      # re-sorted to [v0, v1] based on their slice info offsets.
      pv_0 = variables.PartitionedVariable(
          name="two_vars",
          shape=[2],
          dtype=v0.dtype,
          variable_list=[v0, v1],
          partitions=partitions)

      pv_1 = variables.PartitionedVariable(
          name="two_vars",
          shape=[2],
          dtype=v0.dtype,
          variable_list=[v2, v3],
          partitions=partitions)

      deltas_a = constant_op.constant([1.0, 2.0])
      deltas_b = constant_op.constant([3.0, 4.0])
      ones = array_ops.ones([2])
      plus_delta = pv_0.assign_add(deltas_a)
      minus_delta = pv_0.assign_sub(deltas_b)
      assign_ones = pv_0.assign(ones)

      c_0 = constant_op.constant([2.0])
      c_1 = constant_op.constant([3.0])
      assign_list = pv_1.assign([c_0, c_1])
      assign_part_value = pv_1.assign_add(assign_ones)
      assign_part_var = pv_1.assign_sub(pv_0)
      self.evaluate(variables.global_variables_initializer())

      self.assertEqual([1.0], self.evaluate(plus_delta[0]))
      self.assertEqual([1.0], self.evaluate(v0))
      self.assertEqual([3.0], self.evaluate(plus_delta[1]))
      self.assertEqual([3.0], self.evaluate(v1))

      self.assertEqual([-2.0], self.evaluate(minus_delta[0]))
      self.assertEqual([-2.0], self.evaluate(v0))
      self.assertEqual([-1.0], self.evaluate(minus_delta[1]))
      self.assertEqual([-1.0], self.evaluate(v1))

      self.assertEqual([1.0], self.evaluate(assign_ones[0]))
      self.assertEqual([1.0], self.evaluate(v0))
      self.assertEqual([1.0], self.evaluate(assign_ones[1]))
      self.assertEqual([1.0], self.evaluate(v1))

      self.assertEqual([2.0], self.evaluate(assign_list[0]))
      self.assertEqual([2.0], self.evaluate(v2))
      self.assertEqual([3.0], self.evaluate(assign_list[1]))
      self.assertEqual([3.0], self.evaluate(v3))

      self.assertEqual([3.0], self.evaluate(assign_part_value[0]))
      self.assertEqual([3.0], self.evaluate(v2))
      self.assertEqual([4.0], self.evaluate(assign_part_value[1]))
      self.assertEqual([4.0], self.evaluate(v3))

      self.assertEqual([2.0], self.evaluate(assign_part_var[0]))
      self.assertEqual([2.0], self.evaluate(v2))
      self.assertEqual([3.0], self.evaluate(assign_part_var[1]))
      self.assertEqual([3.0], self.evaluate(v3))


if __name__ == "__main__":
  test.main()
