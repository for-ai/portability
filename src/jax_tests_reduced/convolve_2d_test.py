# Copyright 2020 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unittest
from functools import partial

import jax.numpy as jnp
import jax.scipy.signal as jsp_signal
import numpy as np
import scipy.signal as osp_signal
from absl.testing import absltest
from jax import lax
from jax._src import dtypes
from jax._src import test_util as jtu
from jax.config import config

from ..utils.timer_wrapper import jax_op_timer, partial_timed

config.parse_flags_with_absl()

onedim_shapes = [(1,), (2,), (5,), (10,)]
twodim_shapes = [(1, 1), (2, 2), (2, 3), (3, 4), (4, 4)]
threedim_shapes = [(2, 2, 2), (3, 3, 2), (4, 4, 2), (5, 5, 2)]
stft_test_shapes = [
    # (input_shape, nperseg, noverlap, axis)
    ((50,), 17, 5, -1),
    ((2, 13), 7, 0, -1),
    ((3, 17, 2), 9, 3, 1),
    ((2, 3, 389, 5), 17, 13, 2),
    ((2, 1, 133, 3), 17, 13, -2),
    ((3, 7), 1, 0, 1),
]
csd_test_shapes = [
    # (x_input_shape, y_input_shape, nperseg, noverlap, axis)
    ((50,), (13,), 17, 5, -1),
    ((2, 13), (2, 13), 7, 0, -1),
    ((3, 17, 2), (3, 12, 2), 9, 3, 1),
]
welch_test_shapes = stft_test_shapes
istft_test_shapes = [
    # (input_shape, nperseg, noverlap, timeaxis, freqaxis)
    ((3, 2, 64, 31), 100, 75, -1, -2),
    ((17, 8, 5), 13, 7, 0, 1),
    ((65, 24), 24, 7, -2, -1),
]


default_dtypes = jtu.dtypes.floating + jtu.dtypes.integer + jtu.dtypes.complex
_TPU_FFT_TOL = 0.15


def _real_dtype(dtype):
    return jnp.finfo(dtypes.to_inexact_dtype(dtype)).dtype


def _complex_dtype(dtype):
    return dtypes.to_complex_dtype(dtype)


class LaxBackedScipySignalTests(jtu.JaxTestCase):
    """Tests for LAX-backed scipy.stats implementations"""

    @jtu.sample_product(
        mode=["full", "same", "valid"],
        op=["convolve2d"],
        dtype=default_dtypes,
        xshape=twodim_shapes,
        yshape=twodim_shapes,
    )
    def testConvolutions2D(self, xshape, yshape, dtype, mode, op):
        
        jsp_op = getattr(jsp_signal, op)
        osp_op = getattr(osp_signal, op)
        rng = jtu.rand_default(self.rng())
        args_maker = lambda: [rng(xshape, dtype), rng(yshape, dtype)]
        osp_fun = partial(osp_op, mode=mode)
        jsp_fun = partial_timed(jsp_op, mode=mode, precision=lax.Precision.HIGHEST)
        tol = {
            np.float16: 1e-2,
            np.float32: 1e-2,
            np.float64: 1e-12,
            np.complex64: 1e-2,
            np.complex128: 1e-12,
        }
        self._CheckAgainstNumpy(
            osp_fun, jsp_fun, args_maker, check_dtypes=False, tol=tol
        )
        self._CompileAndCheck(jsp_fun, args_maker, rtol=tol, atol=tol)
