from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import full_type_pb2
from tensorflow.python import tf2
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class DivAndModTest(test_util.TensorFlowTestCase):
    def testDivideType(self):
        a = array_ops.constant([2], dtype=dtypes.int32)
        # Since __future__.division is effect, we should always upgrade to float64
        b = math_ops.divide(a, 1)
        self.assertEqual(b.dtype, dtypes.float64)
        self.assertEqual(2.0, self.evaluate(b))
        c = math_ops.divide(a, 4)
        self.assertEqual(c.dtype, dtypes.float64)
        self.assertEqual(0.5, self.evaluate(c))

    def testComplexDiv(self):
        foo = array_ops.constant([1. + 3.j])
        _ = math_ops.divide(foo, 1.)
        _ = math_ops.div(foo, 2.)

    def testFloorDivGrad(self):
        a = variables.Variable(2.)
        b = variables.Variable(4.)
        input_vars = [a, b]
        self.evaluate(variables.global_variables_initializer())
        if context.executing_eagerly():
            # TDOO(rmlarsen): Is there a more compact way of
            # writing this for multiple expressions?
            with backprop.GradientTape() as tape:
                tape.watch(input_vars)
                c_grad0 = tape.gradient(math_ops.divide(a, b), input_vars)
            with backprop.GradientTape() as tape:
                tape.watch(input_vars)
                c_grad1 = tape.gradient(math_ops.div(a, b), input_vars)
            with backprop.GradientTape() as tape:
                tape.watch(input_vars)
                c_grad2 = tape.gradient(math_ops.floordiv(a, b), input_vars)
        else:
            c_grad0 = gradients.gradients(math_ops.divide(a, b), input_vars)
            c_grad1 = gradients.gradients(math_ops.div(a, b), input_vars)
            c_grad2 = gradients.gradients(math_ops.floordiv(a, b), input_vars)
        self.assertAllEqual([self.evaluate(x) for x in c_grad0], [.25, -.125])
        self.assertAllEqual([self.evaluate(x) for x in c_grad1], [.25, -.125])
        self.assertAllEqual(
            [None if x is None else self.evaluate(x) for x in c_grad2],
            [None, None])


if __name__ == "__main__":
    googletest.main()
