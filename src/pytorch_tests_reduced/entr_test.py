# Owner(s): ["oncall: jit"]

import torch

from torch.testing._internal.jit_utils import JitTestCase
from typing import List
from ..utils.pytorch_device_decorators import onlyNativeDeviceTypes, onlyAcceleratedDeviceTypes, instantiate_device_type_tests
from ..utils.timer_wrapper import pytorch_op_timer

class TestAutodiffJit(JitTestCase):
    def test_requires_grad_outputs_profiled_twice(self, device):
        # the value "r" is used twice, by gammaln and by entr, so it is profiled twice.
        # So during autodiff graph formation the profile nodes are unmerged because
        # they are aliasing. Then the DifferentiableGraph doesn't have a profile
        # node on the output. The requires_grad info should then be added onto the
        # output value (otherwise autodiff will make the output require_grad).
        # Note: this relies on gammaln and entr not having autodiff implementations.
        def fn(a, b, c):
            r = a.relu().relu()
            with pytorch_op_timer():
                test_1 = torch.special.entr(r)
            return  test_1

        fn_s = torch.jit.script(fn)

        a = torch.rand((10, 10), requires_grad=False, device=device)
        b = torch.rand((10, 10), requires_grad=False, device=device)
        c = torch.rand((10, 10), requires_grad=True, device=device)

        for i in range(4):
            y_s = fn_s(a, b, c)
            y= fn(a, b, c)

            # self.assertEqual(x_s.requires_grad, x.requires_grad)
            self.assertEqual(y_s.requires_grad, y.requires_grad)
            # self.assertEqual(z_s.requires_grad, z.requires_grad)

instantiate_device_type_tests(TestAutodiffJit, globals())