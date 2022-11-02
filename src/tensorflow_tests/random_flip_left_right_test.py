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


class RGBToHSVTest(test_util.TensorFlowTestCase):

  def testBatch(self):
    # Build an arbitrary RGB image
    np.random.seed(7)
    batch_size = 5
    shape = (batch_size, 2, 7, 3)

    for nptype in [np.float32, np.float64]:
      inp = np.random.rand(*shape).astype(nptype)

      # Convert to HSV and back, as a batch and individually
      with self.cached_session():
        batch0 = constant_op.constant(inp)
        batch1 = image_ops.rgb_to_hsv(batch0)
        batch2 = image_ops.hsv_to_rgb(batch1)
        split0 = array_ops.unstack(batch0)
        split1 = list(map(image_ops.rgb_to_hsv, split0))
        split2 = list(map(image_ops.hsv_to_rgb, split1))
        join1 = array_ops.stack(split1)
        join2 = array_ops.stack(split2)
        batch1, batch2, join1, join2 = self.evaluate(
            [batch1, batch2, join1, join2])

      # Verify that processing batch elements together is the same as separate
      self.assertAllClose(batch1, join1)
      self.assertAllClose(batch2, join2)
      self.assertAllClose(batch2, inp)

  def testRGBToHSVRoundTrip(self):
    data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    for nptype in [np.float32, np.float64]:
      rgb_np = np.array(data, dtype=nptype).reshape([2, 2, 3]) / 255.
      with self.cached_session():
        hsv = image_ops.rgb_to_hsv(rgb_np)
        rgb = image_ops.hsv_to_rgb(hsv)
        rgb_tf = self.evaluate(rgb)
      self.assertAllClose(rgb_tf, rgb_np)

  def testRGBToHSVDataTypes(self):
    # Test case for GitHub issue 54855.
    data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    for dtype in [
        dtypes.float32, dtypes.float64, dtypes.float16, dtypes.bfloat16
    ]:
      with self.cached_session(use_gpu=False):
        rgb = math_ops.cast(
            np.array(data, np.float32).reshape([2, 2, 3]) / 255., dtype=dtype)
        hsv = image_ops.rgb_to_hsv(rgb)
        val = image_ops.hsv_to_rgb(hsv)
        out = self.evaluate(val)
        self.assertAllClose(rgb, out, atol=1e-2)


class RGBToYIQTest(test_util.TensorFlowTestCase):

  @test_util.run_without_tensor_float_32(
      "Calls rgb_to_yiq and yiq_to_rgb, which use matmul")
  def testBatch(self):
    # Build an arbitrary RGB image
    np.random.seed(7)
    batch_size = 5
    shape = (batch_size, 2, 7, 3)

    for nptype in [np.float32, np.float64]:
      inp = np.random.rand(*shape).astype(nptype)

      # Convert to YIQ and back, as a batch and individually
      with self.cached_session():
        batch0 = constant_op.constant(inp)
        batch1 = image_ops.rgb_to_yiq(batch0)
        batch2 = image_ops.yiq_to_rgb(batch1)
        split0 = array_ops.unstack(batch0)
        split1 = list(map(image_ops.rgb_to_yiq, split0))
        split2 = list(map(image_ops.yiq_to_rgb, split1))
        join1 = array_ops.stack(split1)
        join2 = array_ops.stack(split2)
        batch1, batch2, join1, join2 = self.evaluate(
            [batch1, batch2, join1, join2])

      # Verify that processing batch elements together is the same as separate
      self.assertAllClose(batch1, join1, rtol=1e-4, atol=1e-4)
      self.assertAllClose(batch2, join2, rtol=1e-4, atol=1e-4)
      self.assertAllClose(batch2, inp, rtol=1e-4, atol=1e-4)


class RGBToYUVTest(test_util.TensorFlowTestCase):

  @test_util.run_without_tensor_float_32(
      "Calls rgb_to_yuv and yuv_to_rgb, which use matmul")
  def testBatch(self):
    # Build an arbitrary RGB image
    np.random.seed(7)
    batch_size = 5
    shape = (batch_size, 2, 7, 3)

    for nptype in [np.float32, np.float64]:
      inp = np.random.rand(*shape).astype(nptype)

      # Convert to YUV and back, as a batch and individually
      with self.cached_session():
        batch0 = constant_op.constant(inp)
        batch1 = image_ops.rgb_to_yuv(batch0)
        batch2 = image_ops.yuv_to_rgb(batch1)
        split0 = array_ops.unstack(batch0)
        split1 = list(map(image_ops.rgb_to_yuv, split0))
        split2 = list(map(image_ops.yuv_to_rgb, split1))
        join1 = array_ops.stack(split1)
        join2 = array_ops.stack(split2)
        batch1, batch2, join1, join2 = self.evaluate(
            [batch1, batch2, join1, join2])

      # Verify that processing batch elements together is the same as separate
      self.assertAllClose(batch1, join1, rtol=1e-4, atol=1e-4)
      self.assertAllClose(batch2, join2, rtol=1e-4, atol=1e-4)
      self.assertAllClose(batch2, inp, rtol=1e-4, atol=1e-4)


class GrayscaleToRGBTest(test_util.TensorFlowTestCase):

  def _RGBToGrayscale(self, images):
    is_batch = True
    if len(images.shape) == 3:
      is_batch = False
      images = np.expand_dims(images, axis=0)
    out_shape = images.shape[0:3] + (1,)
    out = np.zeros(shape=out_shape, dtype=np.uint8)
    for batch in range(images.shape[0]):
      for y in range(images.shape[1]):
        for x in range(images.shape[2]):
          red = images[batch, y, x, 0]
          green = images[batch, y, x, 1]
          blue = images[batch, y, x, 2]
          gray = 0.2989 * red + 0.5870 * green + 0.1140 * blue
          out[batch, y, x, 0] = int(gray)
    if not is_batch:
      out = np.squeeze(out, axis=0)
    return out

  def _TestRGBToGrayscale(self, x_np):
    y_np = self._RGBToGrayscale(x_np)

    with self.cached_session():
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.rgb_to_grayscale(x_tf)
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

  def testBasicRGBToGrayscale(self):
    # 4-D input with batch dimension.
    x_np = np.array(
        [[1, 2, 3], [4, 10, 1]], dtype=np.uint8).reshape([1, 1, 2, 3])
    self._TestRGBToGrayscale(x_np)

    # 3-D input with no batch dimension.
    x_np = np.array([[1, 2, 3], [4, 10, 1]], dtype=np.uint8).reshape([1, 2, 3])
    self._TestRGBToGrayscale(x_np)

  def testBasicGrayscaleToRGB(self):
    # 4-D input with batch dimension.
    x_np = np.array([[1, 2]], dtype=np.uint8).reshape([1, 1, 2, 1])
    y_np = np.array(
        [[1, 1, 1], [2, 2, 2]], dtype=np.uint8).reshape([1, 1, 2, 3])

    with self.cached_session():
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.grayscale_to_rgb(x_tf)
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

    # 3-D input with no batch dimension.
    x_np = np.array([[1, 2]], dtype=np.uint8).reshape([1, 2, 1])
    y_np = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.uint8).reshape([1, 2, 3])

    with self.cached_session():
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.grayscale_to_rgb(x_tf)
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

  def testGrayscaleToRGBInputValidation(self):
    # tests whether the grayscale_to_rgb function raises
    # an exception if the input images' last dimension is
    # not of size 1, i.e. the images have shape
    # [batch size, height, width] or [height, width]

    # tests if an exception is raised if a three dimensional
    # input is used, i.e. the images have shape [batch size, height, width]
    with self.cached_session():
      # 3-D input with batch dimension.
      x_np = np.array([[1, 2]], dtype=np.uint8).reshape([1, 1, 2])

      x_tf = constant_op.constant(x_np, shape=x_np.shape)

      # this is the error message we expect the function to raise
      err_msg = "Last dimension of a grayscale image should be size 1"
      with self.assertRaisesRegex(ValueError, err_msg):
        image_ops.grayscale_to_rgb(x_tf)

    # tests if an exception is raised if a two dimensional
    # input is used, i.e. the images have shape [height, width]
    with self.cached_session():
      # 1-D input without batch dimension.
      x_np = np.array([[1, 2]], dtype=np.uint8).reshape([2])

      x_tf = constant_op.constant(x_np, shape=x_np.shape)

      # this is the error message we expect the function to raise
      err_msg = "must be at least two-dimensional"
      with self.assertRaisesRegex(ValueError, err_msg):
        image_ops.grayscale_to_rgb(x_tf)

  def testShapeInference(self):
    # Shape function requires placeholders and a graph.
    with ops.Graph().as_default():
      # Shape inference works and produces expected output where possible
      rgb_shape = [7, None, 19, 3]
      gray_shape = rgb_shape[:-1] + [1]
      with self.cached_session():
        rgb_tf = array_ops.placeholder(dtypes.uint8, shape=rgb_shape)
        gray = image_ops.rgb_to_grayscale(rgb_tf)
        self.assertEqual(gray_shape, gray.get_shape().as_list())

      with self.cached_session():
        gray_tf = array_ops.placeholder(dtypes.uint8, shape=gray_shape)
        rgb = image_ops.grayscale_to_rgb(gray_tf)
        self.assertEqual(rgb_shape, rgb.get_shape().as_list())

      # Shape inference does not break for unknown shapes
      with self.cached_session():
        rgb_tf_unknown = array_ops.placeholder(dtypes.uint8)
        gray_unknown = image_ops.rgb_to_grayscale(rgb_tf_unknown)
        self.assertFalse(gray_unknown.get_shape())

      with self.cached_session():
        gray_tf_unknown = array_ops.placeholder(dtypes.uint8)
        rgb_unknown = image_ops.grayscale_to_rgb(gray_tf_unknown)
        self.assertFalse(rgb_unknown.get_shape())


class AdjustGamma(test_util.TensorFlowTestCase):

  def test_adjust_gamma_less_zero_float32(self):
    """White image should be returned for gamma equal to zero"""
    with self.cached_session():
      x_data = np.random.uniform(0, 1.0, (8, 8))
      x_np = np.array(x_data, dtype=np.float32)

      x = constant_op.constant(x_np, shape=x_np.shape)

      err_msg = "Gamma should be a non-negative real number"
      with self.assertRaisesRegex(
          (ValueError, errors.InvalidArgumentError), err_msg):
        image_ops.adjust_gamma(x, gamma=-1)

  def test_adjust_gamma_less_zero_uint8(self):
    """White image should be returned for gamma equal to zero"""
    with self.cached_session():
      x_data = np.random.uniform(0, 255, (8, 8))
      x_np = np.array(x_data, dtype=np.uint8)

      x = constant_op.constant(x_np, shape=x_np.shape)

      err_msg = "Gamma should be a non-negative real number"
      with self.assertRaisesRegex(
          (ValueError, errors.InvalidArgumentError), err_msg):
        image_ops.adjust_gamma(x, gamma=-1)

  def test_adjust_gamma_less_zero_tensor(self):
    """White image should be returned for gamma equal to zero"""
    with self.cached_session():
      x_data = np.random.uniform(0, 1.0, (8, 8))
      x_np = np.array(x_data, dtype=np.float32)

      x = constant_op.constant(x_np, shape=x_np.shape)
      y = constant_op.constant(-1.0, dtype=dtypes.float32)

      err_msg = "Gamma should be a non-negative real number"
      with self.assertRaisesRegex(
          (ValueError, errors.InvalidArgumentError), err_msg):
        image = image_ops.adjust_gamma(x, gamma=y)
        self.evaluate(image)

  def _test_adjust_gamma_uint8(self, gamma):
    """Verifying the output with expected results for gamma
    correction for uint8 images
    """
    with self.cached_session():
      x_np = np.random.uniform(0, 255, (8, 8)).astype(np.uint8)
      x = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.adjust_gamma(x, gamma=gamma)
      y_tf = np.trunc(self.evaluate(y))

      # calculate gamma correction using numpy
      # firstly, transform uint8 to float representation
      # then perform correction
      y_np = np.power(x_np / 255.0, gamma)
      # convert correct numpy image back to uint8 type
      y_np = np.trunc(np.clip(y_np * 255.5, 0, 255.0))

      self.assertAllClose(y_tf, y_np, 1e-6)

  def _test_adjust_gamma_float32(self, gamma):
    """Verifying the output with expected results for gamma
    correction for float32 images
    """
    with self.cached_session():
      x_np = np.random.uniform(0, 1.0, (8, 8))
      x = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.adjust_gamma(x, gamma=gamma)
      y_tf = self.evaluate(y)

      y_np = np.clip(np.power(x_np, gamma), 0, 1.0)

      self.assertAllClose(y_tf, y_np, 1e-6)

  def test_adjust_gamma_one_float32(self):
    """Same image should be returned for gamma equal to one"""
    self._test_adjust_gamma_float32(1.0)

  def test_adjust_gamma_one_uint8(self):
    self._test_adjust_gamma_uint8(1.0)

  def test_adjust_gamma_zero_uint8(self):
    """White image should be returned for gamma equal
    to zero for uint8 images
    """
    self._test_adjust_gamma_uint8(gamma=0.0)

  def test_adjust_gamma_less_one_uint8(self):
    """Verifying the output with expected results for gamma
    correction with gamma equal to half for uint8 images
    """
    self._test_adjust_gamma_uint8(gamma=0.5)

  def test_adjust_gamma_greater_one_uint8(self):
    """Verifying the output with expected results for gamma
    correction for uint8 images
    """
    self._test_adjust_gamma_uint8(gamma=1.0)

  def test_adjust_gamma_less_one_float32(self):
    """Verifying the output with expected results for gamma
    correction with gamma equal to half for float32 images
    """
    self._test_adjust_gamma_float32(0.5)

  def test_adjust_gamma_greater_one_float32(self):
    """Verifying the output with expected results for gamma
    correction with gamma equal to two for float32 images
    """
    self._test_adjust_gamma_float32(1.0)

  def test_adjust_gamma_zero_float32(self):
    """White image should be returned for gamma equal
    to zero for float32 images
    """
    self._test_adjust_gamma_float32(0.0)


class AdjustHueTest(test_util.TensorFlowTestCase):

  def testAdjustNegativeHue(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    delta = -0.25
    y_data = [0, 13, 1, 54, 226, 59, 8, 234, 150, 255, 39, 1]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.cached_session():
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.adjust_hue(x, delta)
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

  def testAdjustPositiveHue(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    delta = 0.25
    y_data = [13, 0, 11, 226, 54, 221, 234, 8, 92, 1, 217, 255]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.cached_session():
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.adjust_hue(x, delta)
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

  def testBatchAdjustHue(self):
    x_shape = [2, 1, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    delta = 0.25
    y_data = [13, 0, 11, 226, 54, 221, 234, 8, 92, 1, 217, 255]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.cached_session():
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.adjust_hue(x, delta)
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

  def _adjustHueNp(self, x_np, delta_h):
    self.assertEqual(x_np.shape[-1], 3)
    x_v = x_np.reshape([-1, 3])
    y_v = np.ndarray(x_v.shape, dtype=x_v.dtype)
    channel_count = x_v.shape[0]
    for i in range(channel_count):
      r = x_v[i][0]
      g = x_v[i][1]
      b = x_v[i][2]
      h, s, v = colorsys.rgb_to_hsv(r, g, b)
      h += delta_h
      h = math.fmod(h + 10.0, 1.0)
      r, g, b = colorsys.hsv_to_rgb(h, s, v)
      y_v[i][0] = r
      y_v[i][1] = g
      y_v[i][2] = b
    return y_v.reshape(x_np.shape)

  def _adjustHueTf(self, x_np, delta_h):
    with self.cached_session():
      x = constant_op.constant(x_np)
      y = image_ops.adjust_hue(x, delta_h)
      y_tf = self.evaluate(y)
    return y_tf

  def testAdjustRandomHue(self):
    x_shapes = [
        [2, 2, 3],
        [4, 2, 3],
        [2, 4, 3],
        [2, 5, 3],
        [1000, 1, 3],
    ]
    test_styles = [
        "all_random",
        "rg_same",
        "rb_same",
        "gb_same",
        "rgb_same",
    ]
    for x_shape in x_shapes:
      for test_style in test_styles:
        x_np = np.random.rand(*x_shape) * 255.
        delta_h = np.random.rand() * 2.0 - 1.0
        if test_style == "all_random":
          pass
        elif test_style == "rg_same":
          x_np[..., 1] = x_np[..., 0]
        elif test_style == "rb_same":
          x_np[..., 2] = x_np[..., 0]
        elif test_style == "gb_same":
          x_np[..., 2] = x_np[..., 1]
        elif test_style == "rgb_same":
          x_np[..., 1] = x_np[..., 0]
          x_np[..., 2] = x_np[..., 0]
        else:
          raise AssertionError("Invalid test style: %s" % (test_style))
        y_np = self._adjustHueNp(x_np, delta_h)
        y_tf = self._adjustHueTf(x_np, delta_h)
        self.assertAllClose(y_tf, y_np, rtol=2e-5, atol=1e-5)

  def testInvalidShapes(self):
    fused = False
    if not fused:
      # The tests are known to pass with the fused adjust_hue. We will enable
      # them when the fused implementation is the default.
      return
    x_np = np.random.rand(2, 3) * 255.
    delta_h = np.random.rand() * 2.0 - 1.0
    fused = False
    with self.assertRaisesRegex(ValueError, "Shape must be at least rank 3"):
      self._adjustHueTf(x_np, delta_h)
    x_np = np.random.rand(4, 2, 4) * 255.
    delta_h = np.random.rand() * 2.0 - 1.0
    with self.assertRaisesOpError("input must have 3 channels"):
      self._adjustHueTf(x_np, delta_h)

  def testInvalidDeltaValue(self):
    """Delta value must be in the inetrval of [-1,1]."""
    if not context.executing_eagerly():
      self.skipTest("Eager mode only")
    else:
      with self.cached_session():
        x_shape = [2, 2, 3]
        x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
        x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

        x = constant_op.constant(x_np, shape=x_np.shape)

        err_msg = r"delta must be in the interval \[-1, 1\]"
        with self.assertRaisesRegex(
            (ValueError, errors.InvalidArgumentError), err_msg):
          image_ops.adjust_hue(x, delta=1.5)


class FlipImageBenchmark(test.Benchmark):

  def _benchmarkFlipLeftRight(self, device, cpu_count):
    image_shape = [299, 299, 3]
    warmup_rounds = 100
    benchmark_rounds = 1000
    config = config_pb2.ConfigProto()
    if cpu_count is not None:
      config.inter_op_parallelism_threads = 1
      config.intra_op_parallelism_threads = cpu_count
    with session.Session("", graph=ops.Graph(), config=config) as sess:
      with ops.device(device):
        inputs = variables.Variable(
            random_ops.random_uniform(image_shape, dtype=dtypes.float32) * 255,
            trainable=False,
            dtype=dtypes.float32)
        run_op = image_ops.flip_left_right(inputs)
        self.evaluate(variables.global_variables_initializer())
        for i in range(warmup_rounds + benchmark_rounds):
          if i == warmup_rounds:
            start = time.time()
          self.evaluate(run_op)
    end = time.time()
    step_time = (end - start) / benchmark_rounds
    tag = device + "_%s" % (cpu_count if cpu_count is not None else "_all")
    print("benchmarkFlipLeftRight_299_299_3_%s step_time: %.2f us" %
          (tag, step_time * 1e6))
    self.report_benchmark(
        name="benchmarkFlipLeftRight_299_299_3_%s" % (tag),
        iters=benchmark_rounds,
        wall_time=step_time)

  def _benchmarkRandomFlipLeftRight(self, device, cpu_count):
    image_shape = [299, 299, 3]
    warmup_rounds = 100
    benchmark_rounds = 1000
    config = config_pb2.ConfigProto()
    if cpu_count is not None:
      config.inter_op_parallelism_threads = 1
      config.intra_op_parallelism_threads = cpu_count
    with session.Session("", graph=ops.Graph(), config=config) as sess:
      with ops.device(device):
        inputs = variables.Variable(
            random_ops.random_uniform(image_shape, dtype=dtypes.float32) * 255,
            trainable=False,
            dtype=dtypes.float32)
        run_op = image_ops.random_flip_left_right(inputs)
        self.evaluate(variables.global_variables_initializer())
        for i in range(warmup_rounds + benchmark_rounds):
          if i == warmup_rounds:
            start = time.time()
          self.evaluate(run_op)
    end = time.time()
    step_time = (end - start) / benchmark_rounds
    tag = device + "_%s" % (cpu_count if cpu_count is not None else "_all")
    print("benchmarkRandomFlipLeftRight_299_299_3_%s step_time: %.2f us" %
          (tag, step_time * 1e6))
    self.report_benchmark(
        name="benchmarkRandomFlipLeftRight_299_299_3_%s" % (tag),
        iters=benchmark_rounds,
        wall_time=step_time)

  def _benchmarkBatchedRandomFlipLeftRight(self, device, cpu_count):
    image_shape = [16, 299, 299, 3]
    warmup_rounds = 100
    benchmark_rounds = 1000
    config = config_pb2.ConfigProto()
    if cpu_count is not None:
      config.inter_op_parallelism_threads = 1
      config.intra_op_parallelism_threads = cpu_count
    with session.Session("", graph=ops.Graph(), config=config) as sess:
      with ops.device(device):
        inputs = variables.Variable(
            random_ops.random_uniform(image_shape, dtype=dtypes.float32) * 255,
            trainable=False,
            dtype=dtypes.float32)
        run_op = image_ops.random_flip_left_right(inputs)
        self.evaluate(variables.global_variables_initializer())
        for i in range(warmup_rounds + benchmark_rounds):
          if i == warmup_rounds:
            start = time.time()
          self.evaluate(run_op)
    end = time.time()
    step_time = (end - start) / benchmark_rounds
    tag = device + "_%s" % (cpu_count if cpu_count is not None else "_all")
    print("benchmarkBatchedRandomFlipLeftRight_16_299_299_3_%s step_time: "
          "%.2f us" %
          (tag, step_time * 1e6))
    self.report_benchmark(
        name="benchmarkBatchedRandomFlipLeftRight_16_299_299_3_%s" % (tag),
        iters=benchmark_rounds,
        wall_time=step_time)

  def benchmarkFlipLeftRightCpu1(self):
    self._benchmarkFlipLeftRight("/cpu:0", 1)

  def benchmarkFlipLeftRightCpuAll(self):
    self._benchmarkFlipLeftRight("/cpu:0", None)

  def benchmarkFlipLeftRightGpu(self):
    self._benchmarkFlipLeftRight(test.gpu_device_name(), None)

  def benchmarkRandomFlipLeftRightCpu1(self):
    self._benchmarkRandomFlipLeftRight("/cpu:0", 1)

  def benchmarkRandomFlipLeftRightCpuAll(self):
    self._benchmarkRandomFlipLeftRight("/cpu:0", None)

  def benchmarkRandomFlipLeftRightGpu(self):
    self._benchmarkRandomFlipLeftRight(test.gpu_device_name(), None)

  def benchmarkBatchedRandomFlipLeftRightCpu1(self):
    self._benchmarkBatchedRandomFlipLeftRight("/cpu:0", 1)

  def benchmarkBatchedRandomFlipLeftRightCpuAll(self):
    self._benchmarkBatchedRandomFlipLeftRight("/cpu:0", None)

  def benchmarkBatchedRandomFlipLeftRightGpu(self):
    self._benchmarkBatchedRandomFlipLeftRight(test.gpu_device_name(), None)


class AdjustHueBenchmark(test.Benchmark):

  def _benchmarkAdjustHue(self, device, cpu_count):
    image_shape = [299, 299, 3]
    warmup_rounds = 100
    benchmark_rounds = 1000
    config = config_pb2.ConfigProto()
    if cpu_count is not None:
      config.inter_op_parallelism_threads = 1
      config.intra_op_parallelism_threads = cpu_count
    with self.benchmark_session(config=config, device=device) as sess:
      inputs = variables.Variable(
          random_ops.random_uniform(image_shape, dtype=dtypes.float32) * 255,
          trainable=False,
          dtype=dtypes.float32)
      delta = constant_op.constant(0.1, dtype=dtypes.float32)
      outputs = image_ops.adjust_hue(inputs, delta)
      run_op = control_flow_ops.group(outputs)
      self.evaluate(variables.global_variables_initializer())
      for i in range(warmup_rounds + benchmark_rounds):
        if i == warmup_rounds:
          start = time.time()
        self.evaluate(run_op)
    end = time.time()
    step_time = (end - start) / benchmark_rounds
    tag = device + "_%s" % (cpu_count if cpu_count is not None else "_all")
    print("benchmarkAdjustHue_299_299_3_%s step_time: %.2f us" %
          (tag, step_time * 1e6))
    self.report_benchmark(
        name="benchmarkAdjustHue_299_299_3_%s" % (tag),
        iters=benchmark_rounds,
        wall_time=step_time)

  def benchmarkAdjustHueCpu1(self):
    self._benchmarkAdjustHue("/cpu:0", 1)

  def benchmarkAdjustHueCpuAll(self):
    self._benchmarkAdjustHue("/cpu:0", None)

  def benchmarkAdjustHueGpu(self):
    self._benchmarkAdjustHue(test.gpu_device_name(), None)


class AdjustSaturationBenchmark(test.Benchmark):

  def _benchmarkAdjustSaturation(self, device, cpu_count):
    image_shape = [299, 299, 3]
    warmup_rounds = 100
    benchmark_rounds = 1000
    config = config_pb2.ConfigProto()
    if cpu_count is not None:
      config.inter_op_parallelism_threads = 1
      config.intra_op_parallelism_threads = cpu_count
    with self.benchmark_session(config=config, device=device) as sess:
      inputs = variables.Variable(
          random_ops.random_uniform(image_shape, dtype=dtypes.float32) * 255,
          trainable=False,
          dtype=dtypes.float32)
      delta = constant_op.constant(0.1, dtype=dtypes.float32)
      outputs = image_ops.adjust_saturation(inputs, delta)
      run_op = control_flow_ops.group(outputs)
      self.evaluate(variables.global_variables_initializer())
      for _ in range(warmup_rounds):
        self.evaluate(run_op)
      start = time.time()
      for _ in range(benchmark_rounds):
        self.evaluate(run_op)
    end = time.time()
    step_time = (end - start) / benchmark_rounds
    tag = device + "_%s" % (cpu_count if cpu_count is not None else "_all")
    print("benchmarkAdjustSaturation_299_299_3_%s step_time: %.2f us" %
          (tag, step_time * 1e6))
    self.report_benchmark(
        name="benchmarkAdjustSaturation_299_299_3_%s" % (tag),
        iters=benchmark_rounds,
        wall_time=step_time)

  def benchmarkAdjustSaturationCpu1(self):
    self._benchmarkAdjustSaturation("/cpu:0", 1)

  def benchmarkAdjustSaturationCpuAll(self):
    self._benchmarkAdjustSaturation("/cpu:0", None)

  def benchmarkAdjustSaturationGpu(self):
    self._benchmarkAdjustSaturation(test.gpu_device_name(), None)


class ResizeBilinearBenchmark(test.Benchmark):

  def _benchmarkResize(self, image_size, num_channels):
    batch_size = 1
    num_ops = 1000
    img = variables.Variable(
        random_ops.random_normal(
            [batch_size, image_size[0], image_size[1], num_channels]),
        name="img")

    deps = []
    for _ in range(num_ops):
      with ops.control_dependencies(deps):
        resize_op = image_ops.resize_bilinear(
            img, [299, 299], align_corners=False)
        deps = [resize_op]
      benchmark_op = control_flow_ops.group(*deps)

    with self.benchmark_session() as sess:
      self.evaluate(variables.global_variables_initializer())
      results = self.run_op_benchmark(
          sess,
          benchmark_op,
          name=("resize_bilinear_%s_%s_%s" % (image_size[0], image_size[1],
                                              num_channels)))
      print("%s   : %.2f ms/img" %
            (results["name"],
             1000 * results["wall_time"] / (batch_size * num_ops)))

  def benchmarkSimilar3Channel(self):
    self._benchmarkResize((183, 229), 3)

  def benchmarkScaleUp3Channel(self):
    self._benchmarkResize((141, 186), 3)

  def benchmarkScaleDown3Channel(self):
    self._benchmarkResize((749, 603), 3)

  def benchmarkSimilar1Channel(self):
    self._benchmarkResize((183, 229), 1)

  def benchmarkScaleUp1Channel(self):
    self._benchmarkResize((141, 186), 1)

  def benchmarkScaleDown1Channel(self):
    self._benchmarkResize((749, 603), 1)


class ResizeBicubicBenchmark(test.Benchmark):

  def _benchmarkResize(self, image_size, num_channels):
    batch_size = 1
    num_ops = 1000
    img = variables.Variable(
        random_ops.random_normal(
            [batch_size, image_size[0], image_size[1], num_channels]),
        name="img")

    deps = []
    for _ in range(num_ops):
      with ops.control_dependencies(deps):
        resize_op = image_ops.resize_bicubic(
            img, [299, 299], align_corners=False)
        deps = [resize_op]
      benchmark_op = control_flow_ops.group(*deps)

    with self.benchmark_session() as sess:
      self.evaluate(variables.global_variables_initializer())
      results = self.run_op_benchmark(
          sess,
          benchmark_op,
          min_iters=20,
          name=("resize_bicubic_%s_%s_%s" % (image_size[0], image_size[1],
                                             num_channels)))
      print("%s   : %.2f ms/img" %
            (results["name"],
             1000 * results["wall_time"] / (batch_size * num_ops)))

  def benchmarkSimilar3Channel(self):
    self._benchmarkResize((183, 229), 3)

  def benchmarkScaleUp3Channel(self):
    self._benchmarkResize((141, 186), 3)

  def benchmarkScaleDown3Channel(self):
    self._benchmarkResize((749, 603), 3)

  def benchmarkSimilar1Channel(self):
    self._benchmarkResize((183, 229), 1)

  def benchmarkScaleUp1Channel(self):
    self._benchmarkResize((141, 186), 1)

  def benchmarkScaleDown1Channel(self):
    self._benchmarkResize((749, 603), 1)

  def benchmarkSimilar4Channel(self):
    self._benchmarkResize((183, 229), 4)

  def benchmarkScaleUp4Channel(self):
    self._benchmarkResize((141, 186), 4)

  def benchmarkScaleDown4Channel(self):
    self._benchmarkResize((749, 603), 4)


class ResizeAreaBenchmark(test.Benchmark):

  def _benchmarkResize(self, image_size, num_channels):
    batch_size = 1
    num_ops = 1000
    img = variables.Variable(
        random_ops.random_normal(
            [batch_size, image_size[0], image_size[1], num_channels]),
        name="img")

    deps = []
    for _ in range(num_ops):
      with ops.control_dependencies(deps):
        resize_op = image_ops.resize_area(img, [299, 299], align_corners=False)
        deps = [resize_op]
      benchmark_op = control_flow_ops.group(*deps)

    with self.benchmark_session() as sess:
      self.evaluate(variables.global_variables_initializer())
      results = self.run_op_benchmark(
          sess,
          benchmark_op,
          name=("resize_area_%s_%s_%s" % (image_size[0], image_size[1],
                                          num_channels)))
      print("%s   : %.2f ms/img" %
            (results["name"],
             1000 * results["wall_time"] / (batch_size * num_ops)))

  def benchmarkSimilar3Channel(self):
    self._benchmarkResize((183, 229), 3)

  def benchmarkScaleUp3Channel(self):
    self._benchmarkResize((141, 186), 3)

  def benchmarkScaleDown3Channel(self):
    self._benchmarkResize((749, 603), 3)

  def benchmarkSimilar1Channel(self):
    self._benchmarkResize((183, 229), 1)

  def benchmarkScaleUp1Channel(self):
    self._benchmarkResize((141, 186), 1)

  def benchmarkScaleDown1Channel(self):
    self._benchmarkResize((749, 603), 1)


class AdjustSaturationTest(test_util.TensorFlowTestCase):

  def testHalfSaturation(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    saturation_factor = 0.5
    y_data = [6, 9, 13, 140, 180, 226, 135, 121, 234, 172, 255, 128]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.cached_session():
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.adjust_saturation(x, saturation_factor)
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

  def testTwiceSaturation(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    saturation_factor = 2.0
    y_data = [0, 5, 13, 0, 106, 226, 30, 0, 234, 89, 255, 0]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.cached_session():
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.adjust_saturation(x, saturation_factor)
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

  def testBatchSaturation(self):
    x_shape = [2, 1, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    saturation_factor = 0.5
    y_data = [6, 9, 13, 140, 180, 226, 135, 121, 234, 172, 255, 128]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.cached_session():
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.adjust_saturation(x, saturation_factor)
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

  def _adjustSaturationNp(self, x_np, scale):
    self.assertEqual(x_np.shape[-1], 3)
    x_v = x_np.reshape([-1, 3])
    y_v = np.ndarray(x_v.shape, dtype=x_v.dtype)
    channel_count = x_v.shape[0]
    for i in range(channel_count):
      r = x_v[i][0]
      g = x_v[i][1]
      b = x_v[i][2]
      h, s, v = colorsys.rgb_to_hsv(r, g, b)
      s *= scale
      s = min(1.0, max(0.0, s))
      r, g, b = colorsys.hsv_to_rgb(h, s, v)
      y_v[i][0] = r
      y_v[i][1] = g
      y_v[i][2] = b
    return y_v.reshape(x_np.shape)

  def testAdjustRandomSaturation(self):
    x_shapes = [
        [2, 2, 3],
        [4, 2, 3],
        [2, 4, 3],
        [2, 5, 3],
        [1000, 1, 3],
    ]
    test_styles = [
        "all_random",
        "rg_same",
        "rb_same",
        "gb_same",
        "rgb_same",
    ]
    with self.cached_session():
      for x_shape in x_shapes:
        for test_style in test_styles:
          x_np = np.random.rand(*x_shape) * 255.
          scale = np.random.rand()
          if test_style == "all_random":
            pass
          elif test_style == "rg_same":
            x_np[..., 1] = x_np[..., 0]
          elif test_style == "rb_same":
            x_np[..., 2] = x_np[..., 0]
          elif test_style == "gb_same":
            x_np[..., 2] = x_np[..., 1]
          elif test_style == "rgb_same":
            x_np[..., 1] = x_np[..., 0]
            x_np[..., 2] = x_np[..., 0]
          else:
            raise AssertionError("Invalid test style: %s" % (test_style))
          y_baseline = self._adjustSaturationNp(x_np, scale)
          y_fused = self.evaluate(image_ops.adjust_saturation(x_np, scale))
          self.assertAllClose(y_fused, y_baseline, rtol=2e-5, atol=1e-5)


class FlipTransposeRotateTest(test_util.TensorFlowTestCase,
                              parameterized.TestCase):

  def testInvolutionLeftRight(self):
    x_np = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.uint8).reshape([2, 3, 1])
    with self.cached_session():
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.flip_left_right(image_ops.flip_left_right(x_tf))
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, x_np)

  def testInvolutionLeftRightWithBatch(self):
    x_np = np.array(
        [[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]],
        dtype=np.uint8).reshape([2, 2, 3, 1])
    with self.cached_session():
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.flip_left_right(image_ops.flip_left_right(x_tf))
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, x_np)

  def testLeftRight(self):
    x_np = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.uint8).reshape([2, 3, 1])
    y_np = np.array([[3, 2, 1], [3, 2, 1]], dtype=np.uint8).reshape([2, 3, 1])

    with self.cached_session():
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.flip_left_right(x_tf)
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

  def testLeftRightWithBatch(self):
    x_np = np.array(
        [[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]],
        dtype=np.uint8).reshape([2, 2, 3, 1])
    y_np = np.array(
        [[[3, 2, 1], [3, 2, 1]], [[3, 2, 1], [3, 2, 1]]],
        dtype=np.uint8).reshape([2, 2, 3, 1])

    with self.cached_session():
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.flip_left_right(x_tf)
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

  def testRandomFlipLeftRightStateful(self):
    # Test random flip with single seed (stateful).
    with ops.Graph().as_default():
      x_np = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.uint8).reshape([2, 3, 1])
      y_np = np.array([[3, 2, 1], [3, 2, 1]], dtype=np.uint8).reshape([2, 3, 1])
      seed = 42

      with self.cached_session():
        x_tf = constant_op.constant(x_np, shape=x_np.shape)
        y = image_ops.random_flip_left_right(x_tf, seed=seed)
        self.assertTrue(y.op.name.startswith("random_flip_left_right"))

        count_flipped = 0
        count_unflipped = 0
        for _ in range(100):
          y_tf = self.evaluate(y)
          if y_tf[0][0] == 1:
            self.assertAllEqual(y_tf, x_np)
            count_unflipped += 1
          else:
            self.assertAllEqual(y_tf, y_np)
            count_flipped += 1

        # 100 trials
        # Mean: 50
        # Std Dev: ~5
        # Six Sigma: 50 - (5 * 6) = 20
        self.assertGreaterEqual(count_flipped, 20)
        self.assertGreaterEqual(count_unflipped, 20)

  def testRandomFlipLeftRight(self):
    x_np = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.uint8).reshape([2, 3, 1])
    y_np = np.array([[3, 2, 1], [3, 2, 1]], dtype=np.uint8).reshape([2, 3, 1])

    with self.cached_session():
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      count_flipped = 0
      count_unflipped = 0
      for seed in range(100):
        y_tf = self.evaluate(image_ops.random_flip_left_right(x_tf, seed=seed))
        if y_tf[0][0] == 1:
          self.assertAllEqual(y_tf, x_np)
          count_unflipped += 1
        else:
          self.assertAllEqual(y_tf, y_np)
          count_flipped += 1

      self.assertEqual(count_flipped, 45)
      self.assertEqual(count_unflipped, 55)

  # TODO(b/162345082): stateless random op generates different random number
  # with xla_gpu. Update tests such that there is a single ground truth result
  # to test against.
  @parameterized.named_parameters(
      ("_RandomFlipLeftRight", image_ops.stateless_random_flip_left_right),
      ("_RandomFlipUpDown", image_ops.stateless_random_flip_up_down),
  )
  def testRandomFlipStateless(self, func):
    with test_util.use_gpu():
      x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8).reshape([2, 3, 1])
      y_np = np.array([[3, 2, 1], [6, 5, 4]], dtype=np.uint8).reshape([2, 3, 1])
      if "RandomFlipUpDown" in self.id():
        y_np = np.array(
            [[4, 5, 6], [1, 2, 3]], dtype=np.uint8).reshape([2, 3, 1])

      x_tf = constant_op.constant(x_np, shape=x_np.shape)

      iterations = 2
      flip_counts = [None for _ in range(iterations)]
      flip_sequences = ["" for _ in range(iterations)]
      test_seed = (1, 2)
      split_seeds = stateless_random_ops.split(test_seed, 10)
      seeds_list = self.evaluate(split_seeds)
      for i in range(iterations):
        count_flipped = 0
        count_unflipped = 0
        flip_seq = ""
        for seed in seeds_list:
          y_tf = func(x_tf, seed=seed)
          y_tf_eval = self.evaluate(y_tf)
          if y_tf_eval[0][0] == 1:
            self.assertAllEqual(y_tf_eval, x_np)
            count_unflipped += 1
            flip_seq += "U"
          else:
            self.assertAllEqual(y_tf_eval, y_np)
            count_flipped += 1
            flip_seq += "F"

        flip_counts[i] = (count_flipped, count_unflipped)
        flip_sequences[i] = flip_seq

      # Verify that results are deterministic.
      for i in range(1, iterations):
        self.assertAllEqual(flip_counts[0], flip_counts[i])
        self.assertAllEqual(flip_sequences[0], flip_sequences[i])

  # TODO(b/162345082): stateless random op generates different random number
  # with xla_gpu. Update tests such that there is a single ground truth result
  # to test against.
  @parameterized.named_parameters(
      ("_RandomFlipLeftRight", image_ops.stateless_random_flip_left_right),
      ("_RandomFlipUpDown", image_ops.stateless_random_flip_up_down)
  )
  def testRandomFlipStatelessWithBatch(self, func):
    with test_util.use_gpu():
      batch_size = 16

      # create single item of test data
      x_np_raw = np.array(
          [[1, 2, 3], [4, 5, 6]], dtype=np.uint8).reshape([1, 2, 3, 1])
      y_np_raw = np.array(
          [[3, 2, 1], [6, 5, 4]], dtype=np.uint8).reshape([1, 2, 3, 1])
      if "RandomFlipUpDown" in self.id():
        y_np_raw = np.array(
            [[4, 5, 6], [1, 2, 3]], dtype=np.uint8).reshape([1, 2, 3, 1])

      # create batched test data
      x_np = np.vstack([x_np_raw for _ in range(batch_size)])
      y_np = np.vstack([y_np_raw for _ in range(batch_size)])

      x_tf = constant_op.constant(x_np, shape=x_np.shape)

      iterations = 2
      flip_counts = [None for _ in range(iterations)]
      flip_sequences = ["" for _ in range(iterations)]
      test_seed = (1, 2)
      split_seeds = stateless_random_ops.split(test_seed, 10)
      seeds_list = self.evaluate(split_seeds)
      for i in range(iterations):
        count_flipped = 0
        count_unflipped = 0
        flip_seq = ""
        for seed in seeds_list:
          y_tf = func(x_tf, seed=seed)
          y_tf_eval = self.evaluate(y_tf)
          for j in range(batch_size):
            if y_tf_eval[j][0][0] == 1:
              self.assertAllEqual(y_tf_eval[j], x_np[j])
              count_unflipped += 1
              flip_seq += "U"
            else:
              self.assertAllEqual(y_tf_eval[j], y_np[j])
              count_flipped += 1
              flip_seq += "F"

        flip_counts[i] = (count_flipped, count_unflipped)
        flip_sequences[i] = flip_seq

      for i in range(1, iterations):
        self.assertAllEqual(flip_counts[0], flip_counts[i])
        self.assertAllEqual(flip_sequences[0], flip_sequences[i])

  def testRandomFlipLeftRightWithBatch(self):
    batch_size = 16
    seed = 42

    # create single item of test data
    x_np_raw = np.array(
        [[1, 2, 3], [1, 2, 3]], dtype=np.uint8
    ).reshape([1, 2, 3, 1])
    y_np_raw = np.array(
        [[3, 2, 1], [3, 2, 1]], dtype=np.uint8
    ).reshape([1, 2, 3, 1])

    # create batched test data
    x_np = np.vstack([x_np_raw for _ in range(batch_size)])
    y_np = np.vstack([y_np_raw for _ in range(batch_size)])

    with self.cached_session():
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      count_flipped = 0
      count_unflipped = 0
      for seed in range(100):
        y_tf = self.evaluate(image_ops.random_flip_left_right(x_tf, seed=seed))

        # check every element of the batch
        for i in range(batch_size):
          if y_tf[i][0][0] == 1:
            self.assertAllEqual(y_tf[i], x_np[i])
            count_unflipped += 1
          else:
            self.assertAllEqual(y_tf[i], y_np[i])
            count_flipped += 1

      self.assertEqual(count_flipped, 772)
      self.assertEqual(count_unflipped, 828)

  def testInvolutionUpDown(self):
    x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8).reshape([2, 3, 1])

    with self.cached_session():
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.flip_up_down(image_ops.flip_up_down(x_tf))
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, x_np)

  def testInvolutionUpDownWithBatch(self):
    x_np = np.array(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
        dtype=np.uint8).reshape([2, 2, 3, 1])

    with self.cached_session():
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.flip_up_down(image_ops.flip_up_down(x_tf))
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, x_np)

  def testUpDown(self):
    x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8).reshape([2, 3, 1])
    y_np = np.array([[4, 5, 6], [1, 2, 3]], dtype=np.uint8).reshape([2, 3, 1])

    with self.cached_session():
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.flip_up_down(x_tf)
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

  def testUpDownWithBatch(self):
    x_np = np.array(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
        dtype=np.uint8).reshape([2, 2, 3, 1])
    y_np = np.array(
        [[[4, 5, 6], [1, 2, 3]], [[10, 11, 12], [7, 8, 9]]],
        dtype=np.uint8).reshape([2, 2, 3, 1])

    with self.cached_session():
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.flip_up_down(x_tf)
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

  def testRandomFlipUpDownStateful(self):
    # Test random flip with single seed (stateful).
    with ops.Graph().as_default():
      x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8).reshape([2, 3, 1])
      y_np = np.array([[4, 5, 6], [1, 2, 3]], dtype=np.uint8).reshape([2, 3, 1])
      seed = 42

      with self.cached_session():
        x_tf = constant_op.constant(x_np, shape=x_np.shape)
        y = image_ops.random_flip_up_down(x_tf, seed=seed)
        self.assertTrue(y.op.name.startswith("random_flip_up_down"))
        count_flipped = 0
        count_unflipped = 0
        for _ in range(100):
          y_tf = self.evaluate(y)
          if y_tf[0][0] == 1:
            self.assertAllEqual(y_tf, x_np)
            count_unflipped += 1
          else:
            self.assertAllEqual(y_tf, y_np)
            count_flipped += 1

        # 100 trials
        # Mean: 50
        # Std Dev: ~5
        # Six Sigma: 50 - (5 * 6) = 20
        self.assertGreaterEqual(count_flipped, 20)
        self.assertGreaterEqual(count_unflipped, 20)

  def testRandomFlipUpDown(self):
    x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8).reshape([2, 3, 1])
    y_np = np.array([[4, 5, 6], [1, 2, 3]], dtype=np.uint8).reshape([2, 3, 1])

    with self.cached_session():
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      count_flipped = 0
      count_unflipped = 0
      for seed in range(100):
        y_tf = self.evaluate(image_ops.random_flip_up_down(x_tf, seed=seed))
        if y_tf[0][0] == 1:
          self.assertAllEqual(y_tf, x_np)
          count_unflipped += 1
        else:
          self.assertAllEqual(y_tf, y_np)
          count_flipped += 1

      self.assertEqual(count_flipped, 45)
      self.assertEqual(count_unflipped, 55)

  def testRandomFlipUpDownWithBatch(self):
    batch_size = 16
    seed = 42

    # create single item of test data
    x_np_raw = np.array(
        [[1, 2, 3], [4, 5, 6]], dtype=np.uint8
    ).reshape([1, 2, 3, 1])
    y_np_raw = np.array(
        [[4, 5, 6], [1, 2, 3]], dtype=np.uint8
    ).reshape([1, 2, 3, 1])

    # create batched test data
    x_np = np.vstack([x_np_raw for _ in range(batch_size)])
    y_np = np.vstack([y_np_raw for _ in range(batch_size)])

    with self.cached_session():
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      count_flipped = 0
      count_unflipped = 0
      for seed in range(100):
        y_tf = self.evaluate(image_ops.random_flip_up_down(x_tf, seed=seed))

        # check every element of the batch
        for i in range(batch_size):
          if y_tf[i][0][0] == 1:
            self.assertAllEqual(y_tf[i], x_np[i])
            count_unflipped += 1
          else:
            self.assertAllEqual(y_tf[i], y_np[i])
            count_flipped += 1

      self.assertEqual(count_flipped, 772)
      self.assertEqual(count_unflipped, 828)

  def testInvolutionTranspose(self):
    x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8).reshape([2, 3, 1])

    with self.cached_session():
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.transpose(image_ops.transpose(x_tf))
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, x_np)

  def testInvolutionTransposeWithBatch(self):
    x_np = np.array(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
        dtype=np.uint8).reshape([2, 2, 3, 1])

    with self.cached_session():
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.transpose(image_ops.transpose(x_tf))
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, x_np)

  def testTranspose(self):
    x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8).reshape([2, 3, 1])
    y_np = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.uint8).reshape([3, 2, 1])

    with self.cached_session():
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.transpose(x_tf)
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

  def testTransposeWithBatch(self):
    x_np = np.array(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
        dtype=np.uint8).reshape([2, 2, 3, 1])

    y_np = np.array(
        [[[1, 4], [2, 5], [3, 6]], [[7, 10], [8, 11], [9, 12]]],
        dtype=np.uint8).reshape([2, 3, 2, 1])

    with self.cached_session():
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.transpose(x_tf)
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

  def testPartialShapes(self):
    # Shape function requires placeholders and a graph.
    with ops.Graph().as_default():
      p_unknown_rank = array_ops.placeholder(dtypes.uint8)
      p_unknown_dims_3 = array_ops.placeholder(
          dtypes.uint8, shape=[None, None, None])
      p_unknown_dims_4 = array_ops.placeholder(
          dtypes.uint8, shape=[None, None, None, None])
      p_unknown_width = array_ops.placeholder(dtypes.uint8, shape=[64, None, 3])
      p_unknown_batch = array_ops.placeholder(
          dtypes.uint8, shape=[None, 64, 64, 3])
      p_wrong_rank = array_ops.placeholder(dtypes.uint8, shape=[None, None])
      p_zero_dim = array_ops.placeholder(dtypes.uint8, shape=[64, 0, 3])

      #Ops that support 3D input
      for op in [
          image_ops.flip_left_right, image_ops.flip_up_down,
          image_ops.random_flip_left_right, image_ops.random_flip_up_down,
          image_ops.transpose, image_ops.rot90
      ]:
        transformed_unknown_rank = op(p_unknown_rank)
        self.assertIsNone(transformed_unknown_rank.get_shape().ndims)
        transformed_unknown_dims_3 = op(p_unknown_dims_3)
        self.assertEqual(3, transformed_unknown_dims_3.get_shape().ndims)
        transformed_unknown_width = op(p_unknown_width)
        self.assertEqual(3, transformed_unknown_width.get_shape().ndims)

        with self.assertRaisesRegex(ValueError, "must be > 0"):
          op(p_zero_dim)

      #Ops that support 4D input
      for op in [
          image_ops.flip_left_right, image_ops.flip_up_down,
          image_ops.random_flip_left_right, image_ops.random_flip_up_down,
          image_ops.transpose, image_ops.rot90
      ]:
        transformed_unknown_dims_4 = op(p_unknown_dims_4)
        self.assertEqual(4, transformed_unknown_dims_4.get_shape().ndims)
        transformed_unknown_batch = op(p_unknown_batch)
        self.assertEqual(4, transformed_unknown_batch.get_shape().ndims)
        with self.assertRaisesRegex(ValueError,
                                    "must be at least three-dimensional"):
          op(p_wrong_rank)