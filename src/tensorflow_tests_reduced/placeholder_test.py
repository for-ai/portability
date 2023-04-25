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
"""Tests for fft operations."""

import itertools
import unittest

from absl.testing import parameterized
import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_spectral_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.platform import test
from ..utils.timer_wrapper import tensorflow_op_timer

VALID_FFT_RANKS = (1, 2, 3)


# TODO(rjryan): Investigate precision issues. We should be able to achieve
# better tolerances, at least for the complex128 tests.
class BaseFFTOpsTest(test.TestCase):

  def _compare(self, x, rank, fft_length=None, use_placeholder=False,
               rtol=1e-4, atol=1e-4):
    self._compare_forward(x, rank, fft_length, use_placeholder, rtol, atol)
    self._compare_backward(x, rank, fft_length, use_placeholder, rtol, atol)

  def _compare_forward(self, x, rank, fft_length=None, use_placeholder=False,
                       rtol=1e-4, atol=1e-4):
    x_np = self._np_fft(x, rank, fft_length)
    if use_placeholder:
      timer = tensorflow_op_timer()
      with timer:
        x_ph = array_ops.placeholder(dtype=dtypes.as_dtype(x.dtype))
        timer.gen.send(x_ph)
      x_tf = self._tf_fft(x_ph, rank, fft_length, feed_dict={x_ph: x})
    else:
      x_tf = self._tf_fft(x, rank, fft_length)

    self.assertAllClose(x_np, x_tf, rtol=rtol, atol=atol)

  def _compare_backward(self, x, rank, fft_length=None, use_placeholder=False,
                        rtol=1e-4, atol=1e-4):
    x_np = self._np_ifft(x, rank, fft_length)
    if use_placeholder:
      timer = tensorflow_op_timer()
      with timer:
        x_ph = array_ops.placeholder(dtype=dtypes.as_dtype(x.dtype))
        timer.gen.send(x_ph)
      x_tf = self._tf_ifft(x_ph, rank, fft_length, feed_dict={x_ph: x})
    else:
      x_tf = self._tf_ifft(x, rank, fft_length)

    self.assertAllClose(x_np, x_tf, rtol=rtol, atol=atol)

  def _check_memory_fail(self, x, rank):
    config = config_pb2.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1e-2
    with self.cached_session(config=config, force_gpu=True):
      self._tf_fft(x, rank, fft_length=None)

  def _check_grad_complex(self, func, x, y, result_is_complex=True,
                          rtol=1e-2, atol=1e-2):
    with self.cached_session():

      def f(inx, iny):
        inx.set_shape(x.shape)
        iny.set_shape(y.shape)
        # func is a forward or inverse, real or complex, batched or unbatched
        # FFT function with a complex input.
        z = func(math_ops.complex(inx, iny))
        # loss = sum(|z|^2)
        loss = math_ops.reduce_sum(math_ops.real(z * math_ops.conj(z)))
        return loss

      ((x_jacob_t, y_jacob_t), (x_jacob_n, y_jacob_n)) = (
          gradient_checker_v2.compute_gradient(f, [x, y], delta=1e-2))

    self.assertAllClose(x_jacob_t, x_jacob_n, rtol=rtol, atol=atol)
    self.assertAllClose(y_jacob_t, y_jacob_n, rtol=rtol, atol=atol)

  def _check_grad_real(self, func, x, rtol=1e-2, atol=1e-2):
    def f(inx):
      inx.set_shape(x.shape)
      # func is a forward RFFT function (batched or unbatched).
      z = func(inx)
      # loss = sum(|z|^2)
      loss = math_ops.reduce_sum(math_ops.real(z * math_ops.conj(z)))
      return loss

    (x_jacob_t,), (x_jacob_n,) = gradient_checker_v2.compute_gradient(
        f, [x], delta=1e-2)
    self.assertAllClose(x_jacob_t, x_jacob_n, rtol=rtol, atol=atol)


@test_util.run_all_in_graph_and_eager_modes
class FFTOpsTest(BaseFFTOpsTest, parameterized.TestCase):

  def _tf_fft(self, x, rank, fft_length=None, feed_dict=None):
    # fft_length unused for complex FFTs.
    with self.cached_session() as sess:
      return sess.run(self._tf_fft_for_rank(rank)(x), feed_dict=feed_dict)

  def _tf_ifft(self, x, rank, fft_length=None, feed_dict=None):
    # fft_length unused for complex FFTs.
    with self.cached_session() as sess:
      return sess.run(self._tf_ifft_for_rank(rank)(x), feed_dict=feed_dict)

  def _np_fft(self, x, rank, fft_length=None):
    if rank == 1:
      return np.fft.fft2(x, s=fft_length, axes=(-1,))
    elif rank == 2:
      return np.fft.fft2(x, s=fft_length, axes=(-2, -1))
    elif rank == 3:
      return np.fft.fft2(x, s=fft_length, axes=(-3, -2, -1))
    else:
      raise ValueError("invalid rank")

  def _np_ifft(self, x, rank, fft_length=None):
    if rank == 1:
      return np.fft.ifft2(x, s=fft_length, axes=(-1,))
    elif rank == 2:
      return np.fft.ifft2(x, s=fft_length, axes=(-2, -1))
    elif rank == 3:
      return np.fft.ifft2(x, s=fft_length, axes=(-3, -2, -1))
    else:
      raise ValueError("invalid rank")

  def _tf_fft_for_rank(self, rank):
    if rank == 1:
      return fft_ops.fft
    elif rank == 2:
      return fft_ops.fft2d
    elif rank == 3:
      return fft_ops.fft3d
    else:
      raise ValueError("invalid rank")

  def _tf_ifft_for_rank(self, rank):
    if rank == 1:
      return fft_ops.ifft
    elif rank == 2:
      return fft_ops.ifft2d
    elif rank == 3:
      return fft_ops.ifft3d
    else:
      raise ValueError("invalid rank")

  @parameterized.parameters(itertools.product(
      VALID_FFT_RANKS, range(3), (np.complex64, np.complex128)))
  def test_placeholder(self, rank, extra_dims, np_type):
    if context.executing_eagerly():
      return
    tol = 1e-4 if np_type == np.complex64 else 1e-8
    dims = rank + extra_dims
    self._compare(
        np.mod(np.arange(np.power(4, dims)), 10).reshape(
            (4,) * dims).astype(np_type),
        rank, use_placeholder=True, rtol=tol, atol=tol)

  
  @parameterized.parameters(itertools.product(
      VALID_FFT_RANKS, range(3), (5, 6), (np.float32, np.float64)))
  def test_placeholder(self, rank, extra_dims, size, np_rtype):
    if context.executing_eagerly():
      return
    np_ctype = np.complex64 if np_rtype == np.float32 else np.complex128
    tol = 1e-4 if np_rtype == np.float32 else 1e-8
    dims = rank + extra_dims
    inner_dim = size // 2 + 1
    r2c = np.mod(np.arange(np.power(size, dims)), 10).reshape(
        (size,) * dims)
    fft_length = (size,) * rank
    self._compare_forward(
        r2c.astype(np_rtype),
        rank,
        fft_length,
        use_placeholder=True,
        rtol=tol,
        atol=tol)
    c2r = np.mod(np.arange(np.power(size, dims - 1) * inner_dim),
                 10).reshape((size,) * (dims - 1) + (inner_dim,))
    c2r = self._generate_valid_irfft_input(c2r, np_ctype, r2c, np_rtype, rank,
                                           fft_length)
    self._compare_backward(
        c2r, rank, fft_length, use_placeholder=True, rtol=tol, atol=tol)


  @parameterized.parameters(None, 1, ([1, 2],))
  def test_placeholder(self, axes):
    if context.executing_eagerly():
      return
    timer = tensorflow_op_timer()
    with timer:
      x = array_ops.placeholder(shape=[None, None, None], dtype="float32")
      timer.gen.send(x) 
    y_fftshift = fft_ops.fftshift(x, axes=axes)
    y_ifftshift = fft_ops.ifftshift(x, axes=axes)
    x_np = np.random.rand(16, 256, 256)
    with self.session() as sess:
      y_fftshift_res, y_ifftshift_res = sess.run(
          [y_fftshift, y_ifftshift],
          feed_dict={x: x_np})
    self.assertAllClose(y_fftshift_res, np.fft.fftshift(x_np, axes=axes))
    self.assertAllClose(y_ifftshift_res, np.fft.ifftshift(x_np, axes=axes))

  

if __name__ == "__main__":
  test.main()
