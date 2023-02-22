import contextlib
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_device_type import (
    dtypes, dtypesIfCUDA)
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import run_tests

from ..utils.pytorch_device_decorators import onlyNativeDeviceTypes, onlyAcceleratedDeviceTypes, instantiate_device_type_tests

# TODO: remove this global setting
# NN tests use double as the default dtype
torch.set_default_dtype(torch.double)


class TestNNDeviceType(NNTestCase):
    @dtypes(torch.float)
    @dtypesIfCUDA(torch.half, torch.float)
    def test_transformerencoderlayer_gelu(self, device, dtype):
        # this is a deterministic test for TransformerEncoderLayer with gelu activation
        d_model = 4
        nhead = 2
        dim_feedforward = 16
        dropout = 0.0
        bsz = 2

        atol = 0
        rtol = 1e-5
        if "cuda" in device or "xla" in device:
            atol = 1e-3
            rtol = 1e-2

        def _test(activation, batch_first, training):
            def perm_fn(x):
                return x.transpose(1, 0) if batch_first else x

            model = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                               activation, batch_first=batch_first, device=device, dtype=dtype)
            if not training:
                assert dropout == 0
                model = model.eval()

            # set constant weights of the model
            for idx, p in enumerate(model.parameters()):
                x = p.data
                sz = x.view(-1).size(0)
                shape = x.shape
                x = torch.cos(torch.arange(0, sz).float().view(shape))
                p.data.copy_(x)

            # deterministic input
            encoder_input = torch.tensor(
                [[[20., 30., 40., 50.]]], device=device, dtype=dtype)
            result = model(encoder_input)
            ref_output = torch.tensor(
                [[[2.249815, 0.131006, -0.702199, 0.177868]]], device=device, dtype=dtype)
            torch.testing.assert_close(
                result, ref_output, rtol=rtol, atol=atol)

            # deterministic input
            encoder_input = perm_fn(torch.tensor([[[1., 2., 3., 4.]],
                                                  [[5., 6., 7., 8.]]], device=device, dtype=dtype))
            result = model(encoder_input)
            ref_output = perm_fn(torch.tensor([[[2.264103, 0.121417, -0.696012, 0.159724]],
                                               [[2.264103, 0.121417, -0.696012, 0.159724]]], device=device, dtype=dtype))
            torch.testing.assert_close(
                result, ref_output, rtol=rtol, atol=atol)

            # deterministic input
            encoder_input = perm_fn(torch.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                                  [0.5387, 0.1655, 0.3565, 0.0471]],
                                                  [[0.8335, 0.2799, 0.5031, 0.2947],
                                                  [0.1402, 0.0318, 0.7636, 0.1346]],
                                                  [[0.6333, 0.9344, 0.1376, 0.9938],
                                                  [0.8924, 0.2872, 0.6692, 0.2944]],
                                                  [[0.9897, 0.6915, 0.3154, 0.1733],
                                                  [0.8645, 0.3513, 0.3064, 0.0767]],
                                                  [[0.8117, 0.2366, 0.4838, 0.7881],
                                                  [0.3718, 0.4945, 0.9511, 0.0864]]], device=device, dtype=dtype))
            result = model(encoder_input)
            ref_output = perm_fn(torch.tensor([[[2.42163188, 0.03227153, -0.60714219, -0.05908082],
                                                [2.42151276, 0.03302179, -0.60722523, -0.05762651]],
                                               [[2.41926761, 0.02974034, -0.60879519, -0.0621269],
                                                [2.41626395, 0.03539356, -0.61087842, -0.04978623]],
                                               [[2.42382808, 0.03218872, -0.6055963, -0.06073591],
                                                [2.41983477, 0.03085259, -0.60840145, -0.06046414]],
                                               [[2.42500749, 0.03328855, -0.60476388, -0.0595334],
                                                [2.4237977, 0.03290575, -0.60561789, -0.05940082]],
                                               [[2.41383916, 0.02686345, -0.61256377, -0.06380707],
                                                [2.42000277, 0.03800944, -0.60824798, -0.04754947]]], device=device, dtype=dtype))
            torch.testing.assert_close(
                result, ref_output, rtol=rtol, atol=atol)
        for activation, batch_first, training in product(('gelu', F.gelu, nn.GELU()), (True, False), (True, False)):
            # Fast path requires inference mode.
            if training:
                cm = contextlib.nullcontext()
            else:
                cm = torch.no_grad()
            with cm:
                _test(activation=activation,
                      batch_first=batch_first, training=training)


instantiate_device_type_tests(TestNNDeviceType, globals())
if __name__ == '__main__':
    run_tests()
