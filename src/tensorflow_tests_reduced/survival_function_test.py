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
"""Tests for initializers."""

import importlib
import math

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.ops.distributions import normal as normal_lib
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
from ..utils.timer_wrapper import tensorflow_op_timer

def try_import(name):  # pylint: disable=invalid-name
  module = None
  try:
    module = importlib.import_module(name)
  except ImportError as e:
    tf_logging.warning("Could not import %s: %s" % (name, str(e)))
  return module

stats = try_import("scipy.stats")


class NormalTest(test.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(123)

  def assertAllFinite(self, tensor):
    is_finite = np.isfinite(self.evaluate(tensor))
    all_true = np.ones_like(is_finite, dtype=np.bool_)
    self.assertAllEqual(all_true, is_finite)

  @test_util.run_in_graph_and_eager_modes
  def testSampleLikeArgsGetDistDType(self):
    dist = normal_lib.Normal(0., 1.)
    self.assertEqual(dtypes.float32, dist.dtype)
    for method in ("log_prob", "prob", "log_cdf", "cdf",
                   "log_survival_function", "survival_function", "quantile"):
      self.assertEqual(dtypes.float32, getattr(dist, method)(1).dtype)

  @test_util.run_in_graph_and_eager_modes
  def testNormalSurvivalFunction(self):
    batch_size = 50
    mu = self._rng.randn(batch_size)
    sigma = self._rng.rand(batch_size) + 1.0
    x = np.linspace(-8.0, 8.0, batch_size).astype(np.float64)

    normal = normal_lib.Normal(loc=mu, scale=sigma)
    timer = tensorflow_op_timer()
    with timer:
      sf = normal.survival_function(x)
      timer.gen.send(sf)
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()), sf.get_shape())
    self.assertAllEqual(
        self.evaluate(normal.batch_shape_tensor()),
        self.evaluate(sf).shape)
    self.assertAllEqual(normal.batch_shape, sf.get_shape())
    self.assertAllEqual(normal.batch_shape, self.evaluate(sf).shape)
    if not stats:
      return
    expected_sf = stats.norm(mu, sigma).sf(x)
    self.assertAllClose(expected_sf, self.evaluate(sf), atol=0)

  def testFiniteGradientAtDifficultPoints(self):
    for dtype in [np.float32, np.float64]:
      g = ops.Graph()
      with g.as_default():
        mu = variables.Variable(dtype(0.0))
        sigma = variables.Variable(dtype(1.0))
        dist = normal_lib.Normal(loc=mu, scale=sigma)
        x = np.array([-100., -20., -5., 0., 5., 20., 100.]).astype(dtype)
        for func in [
            dist.survival_function,
            
        ]:
          timer = tensorflow_op_timer()
          with timer:
            value = func(x)
            timer.gen.send(value)
          grads = gradients_impl.gradients(value, [mu, sigma])
          with self.session(graph=g):
            self.evaluate(variables.global_variables_initializer())
            self.assertAllFinite(value)
            self.assertAllFinite(grads[0])
            self.assertAllFinite(grads[1])

if __name__ == "__main__":
  test.main()
