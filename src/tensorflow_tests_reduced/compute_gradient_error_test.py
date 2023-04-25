import functools
import math

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.ops.nn_impl import _compute_sampled_logits
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test as test_lib
from ..utils.timer_wrapper import tensorflow_op_timer


class MomentsTest(test_lib.TestCase):

  def doOutputTest(self,
                   input_shape,
                   moments_axes,
                   tol=1e-4,
                   check_gradients=False):
    for mu in [0.0, 1.0, 1e3]:
      for sigma in [1.0, 0.1]:
        for keep_dims in [True, False]:
          input_values = np.random.rand(*input_shape) * sigma + mu
          expected_mean = np.mean(
              input_values, axis=moments_axes, keepdims=keep_dims)
          expected_var = np.var(
              input_values, axis=moments_axes, keepdims=keep_dims)
          with ops.Graph().as_default() as g:
            with self.session(graph=g) as sess:
              inputs = constant_op.constant(
                  input_values, shape=input_shape, dtype=dtypes.float32)
              mean, variance = nn_impl.moments_v2(
                  inputs, moments_axes, keepdims=keep_dims)

              if check_gradients:
                timer = tensorflow_op_timer()
                with timer:
                  err = gradient_checker.compute_gradient_error(
                    inputs, input_shape, mean, mean.shape.as_list())
                  timer.gen.send(err)
                self.assertLess(err, 1e-3)
                timer = tensorflow_op_timer()
                with timer:
                  err = gradient_checker.compute_gradient_error(
                    inputs, input_shape, variance, variance.shape.as_list())
                  timer.gen.send(err)
                self.assertLess(err, 1e-3)

              # Evaluate.
              [mean, variance] = self.evaluate([mean, variance])
              # Make sure that there are no NaNs
              self.assertFalse(np.isnan(mean).any())
              self.assertFalse(np.isnan(variance).any())
              self.assertAllClose(mean, expected_mean, rtol=tol, atol=tol)
              self.assertAllClose(variance, expected_var, rtol=tol, atol=tol)

  def testOutputAndGradient2DInput0(self):
    self.doOutputTest((10, 10), (0,), check_gradients=True)

  def testOutputAndGradient2DInput01(self):
    self.doOutputTest((10, 10), (0, 1), check_gradients=True)

if __name__ == "__main__":
  test_lib.main()
