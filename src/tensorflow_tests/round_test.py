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
class RoundTest(test_util.TensorFlowTestCase):

  def testRounding(self):
    x = np.arange(-5.0, 5.0, .25)
    for dtype in [np.float32, np.double, np.int32]:
      x_np = np.array(x, dtype=dtype)
      with test_util.device(use_gpu=True):
        x_tf = constant_op.constant(x_np, shape=x_np.shape)
        y_tf = math_ops.round(x_tf)
        y_tf_np = self.evaluate(y_tf)
        y_np = np.round(x_np)
        self.assertAllClose(y_tf_np, y_np, atol=1e-2)


if __name__ == "__main__":
      googletest.main()
