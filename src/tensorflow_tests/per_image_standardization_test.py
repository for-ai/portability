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
"""Tests for tensorflow.ops.image_ops."""

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


class PerImageWhiteningTest(test_util.TensorFlowTestCase,
                            parameterized.TestCase):

  def _NumpyPerImageWhitening(self, x):
    num_pixels = np.prod(x.shape)
    mn = np.mean(x)
    std = np.std(x)
    stddev = max(std, 1.0 / math.sqrt(num_pixels))

    y = x.astype(np.float32)
    y -= mn
    y /= stddev
    return y

  @parameterized.named_parameters([("_int8", np.int8), ("_int16", np.int16),
                                   ("_int32", np.int32), ("_int64", np.int64),
                                   ("_uint8", np.uint8), ("_uint16", np.uint16),
                                   ("_uint32", np.uint32),
                                   ("_uint64", np.uint64),
                                   ("_float32", np.float32)])
  def testBasic(self, data_type):
    x_shape = [13, 9, 3]
    x_np = np.arange(0, np.prod(x_shape), dtype=data_type).reshape(x_shape)
    y_np = self._NumpyPerImageWhitening(x_np)

    with self.cached_session():
      x = constant_op.constant(x_np, dtype=data_type, shape=x_shape)
      y = image_ops.per_image_standardization(x)
      y_tf = self.evaluate(y)
      self.assertAllClose(y_tf, y_np, atol=1e-4)

  def testUniformImage(self):
    im_np = np.ones([19, 19, 3]).astype(np.float32) * 249
    im = constant_op.constant(im_np)
    whiten = image_ops.per_image_standardization(im)
    with self.cached_session():
      whiten_np = self.evaluate(whiten)
      self.assertFalse(np.any(np.isnan(whiten_np)))

  def testBatchWhitening(self):
    imgs_np = np.random.uniform(0., 255., [4, 24, 24, 3])
    whiten_np = [self._NumpyPerImageWhitening(img) for img in imgs_np]
    with self.cached_session():
      imgs = constant_op.constant(imgs_np)
      whiten = image_ops.per_image_standardization(imgs)
      whiten_tf = self.evaluate(whiten)
      for w_tf, w_np in zip(whiten_tf, whiten_np):
        self.assertAllClose(w_tf, w_np, atol=1e-4)


if __name__ == "__main__":
  googletest.main()
