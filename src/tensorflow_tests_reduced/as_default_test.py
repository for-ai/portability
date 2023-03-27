# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

from absl.testing import parameterized

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_grad
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.util import nest


_COS_DERIVATIVES = [math_ops.cos,
                    lambda x: -math_ops.sin(x),
                    lambda x: -math_ops.cos(x),
                    math_ops.sin,
                    math_ops.cos]


class FunctionGradientsTest(test.TestCase, parameterized.TestCase):

    # def setUp(self):
    #     super(FunctionGradientsTest, self).setUp()
    #     cpus = config.list_physical_devices('CPU')
    #     # Set 4 virtual CPUs
    #     config.set_logical_device_configuration(cpus[0], [
    #         context.LogicalDeviceConfiguration(),
    #         context.LogicalDeviceConfiguration(),
    #         context.LogicalDeviceConfiguration(),
    #         context.LogicalDeviceConfiguration()
    #     ])

    def testGraphModeWithGradients(self):
        v = resource_variable_ops.ResourceVariable(1.0, name='v')

        @def_function.function
        def step():
            def inner():
                return v * v

            return backprop.implicit_grad(inner)()[0][0]

        self.assertAllEqual(step(), 2.0)

    def testGraphGradientVariable(self):
        with ops.Graph().as_default(), self.cached_session():
            v = variables.Variable(1.0)

            @def_function.function
            def f():
                return 2.0 * v

            node = f()
            grads, = gradients_impl.gradients(node, v)
            v.initializer.run()
            self.assertAllEqual(grads, 2.0)
            self.assertEqual(grads.shape, v.shape)


if __name__ == '__main__':
    ops.enable_eager_execution()
    test.main()
