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


class FliplrTest(test_util.TensorFlowTestCase):
    def testFliplr(self):
        """Tests against negative SSIM index."""
        step = np.expand_dims(np.arange(0, 256, 16, dtype=np.uint8), axis=0)
        img1 = np.tile(step, (16, 1))
        img2 = np.fliplr(img1)

        img1 = img1.reshape((1, 16, 16, 1))
        img2 = img2.reshape((1, 16, 16, 1))

        ssim = image_ops.ssim(
            constant_op.constant(img1),
            constant_op.constant(img2),
            255,
            filter_size=11,
            filter_sigma=1.5,
            k1=0.01,
            k2=0.03)
        with self.cached_session():
            self.assertLess(self.evaluate(ssim), 0)


if __name__ == "__main__":
    googletest.main()
