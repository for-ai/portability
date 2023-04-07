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
from ..utils.timer_wrapper import tensorflow_op_timer


@test_util.run_all_in_graph_and_eager_modes
class LogSumExpTest(test_util.TensorFlowTestCase):

    def testReduceLogSumExp(self):
        for dtype in [np.float32, np.double]:
            x_np = np.random.rand(5, 5).astype(dtype)
            with tensorflow_op_timer():
                y_tf_np = math_ops.reduce_logsumexp(x_np)
            y_np = np.log(np.sum(np.exp(x_np)))
            self.assertAllClose(y_tf_np, y_np)


if __name__ == "__main__":
    googletest.main()
