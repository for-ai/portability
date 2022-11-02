from absl.testing import parameterized

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops as framework_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops.parallel_for import control_flow_ops as pfor_control_flow_ops
from tensorflow.python.ops.parallel_for.test_util import PForTestCase
from tensorflow.python.platform import test


class RsqrtTest(PForTestCase, parameterized.TestCase):
    def _test_unary_cwise_ops(self, ops, is_complex):
        for op in ops:
            with backprop.GradientTape(persistent=True) as g:
                x = random_ops.random_uniform([3, 5])
                g.watch(x)
                if is_complex:
                    y = random_ops.random_uniform([3, 5])
                    g.watch(y)
                    x = math_ops.complex(x, y)

            # pylint: disable=cell-var-from-loop

            def loop_fn(i):
                with g:
                    y = op(x)
                    x_i = array_ops.gather(x, i)
                    y_i = op(x_i)
                    outputs = [y_i]
                    # Build cross product of loop variant/invariant outputs and gradients.
                    for out in (y, y_i):
                        if out.dtype == dtypes.float32:
                            for output_gradients in (None, out * math_ops.cast(i, out.dtype)):
                                grad = g.gradient(
                                    out, x_i, output_gradients=output_gradients)
                                if grad is not None:
                                    outputs.append(grad)
                return outputs

            # pylint: enable=cell-var-from-loop

            self._test_loop_fn(loop_fn, 3)

    def test_unary_cwise_real_ops_2(self):
        real_ops = [
            math_ops.rsqrt,
        ]
        self._test_unary_cwise_ops(real_ops, False)


if __name__ == "__main__":
    test.main()
