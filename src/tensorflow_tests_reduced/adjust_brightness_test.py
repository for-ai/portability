import colorsys
import contextlib
import functools
import itertools
import math
import os
import time

from absl.testing import parameterized
import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.compat import compat
from tensorflow.python.data.experimental.ops import get_single_element
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import image_ops_impl
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test

from ..utils.timer_wrapper import tensorflow_op_timer

class AdjustBrightnessTest(test_util.TensorFlowTestCase):

    def _testBrightness(self, x_np, y_np, delta, tol=1e-6):
        with self.cached_session():
            x = constant_op.constant(x_np, shape=x_np.shape)
            with tensorflow_op_timer():
                y = image_ops.adjust_brightness(x, delta)
            y_tf = self.evaluate(y)
            self.assertAllClose(y_tf, y_np, tol)

    def testPositiveDeltaUint8(self):
        x_shape = [2, 2, 3]
        x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
        x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

        y_data = [10, 15, 23, 64, 145, 236, 47, 18, 244, 100, 255, 11]
        y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

        self._testBrightness(x_np, y_np, delta=10. / 255.)


if __name__ == "__main__":
    googletest.main()
