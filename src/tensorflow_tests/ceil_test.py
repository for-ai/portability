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


@test_util.run_all_in_graph_and_eager_modes
class MathTest(PForTestCase, parameterized.TestCase):
    def test_unary_cwise_no_grad(self):
        for op in [math_ops.ceil, math_ops.floor, math_ops.logical_not]:
            x = random_ops.random_uniform([3, 5])
            if op == math_ops.logical_not:
                x = x > 0

            # pylint: disable=cell-var-from-loop
            def loop_fn(i):
                return op(array_ops.gather(x, i))

            # pylint: enable=cell-var-from-loop

            self._test_loop_fn(loop_fn, 3)


if __name__ == "__main__":
    test.main()
