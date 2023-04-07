# import copy
# import gc
# import os
# import pickle
# import re

# from absl.testing import parameterized
# import numpy as np

# from tensorflow.core.framework import full_type_pb2
# from tensorflow.core.framework import tensor_pb2
# from tensorflow.python.compat import compat as forward_compat
# from tensorflow.python.eager import backprop
# from tensorflow.python.eager import context
# from tensorflow.python.eager import def_function
# from tensorflow.python.framework import composite_tensor
# from tensorflow.python.framework import constant_op
# from tensorflow.python.framework import cpp_shape_inference_pb2
# from tensorflow.python.framework import dtypes
# from tensorflow.python.framework import errors
# from tensorflow.python.framework import extension_type
# from tensorflow.python.framework import indexed_slices
# from tensorflow.python.framework import memory_checker
# from tensorflow.python.framework import ops
# from tensorflow.python.framework import tensor_shape
# from tensorflow.python.framework import tensor_util
# from tensorflow.python.framework import test_ops
# from tensorflow.python.framework import test_util
# from tensorflow.python.framework import type_spec
# from tensorflow.python.ops import array_ops
# from tensorflow.python.ops import control_flow_ops
# from tensorflow.python.ops import custom_gradient
# from tensorflow.python.ops import gradients_impl
# from tensorflow.python.ops import handle_data_util
# from tensorflow.python.ops import init_ops
# from tensorflow.python.ops import list_ops
# from tensorflow.python.ops import math_ops
# from tensorflow.python.ops import resource_variable_ops
# from tensorflow.python.ops import state_ops
# from tensorflow.python.ops import variable_scope
# from tensorflow.python.ops import variables
# from tensorflow.python.platform import test
# from tensorflow.python.training import momentum
# from tensorflow.python.training import saver
# from tensorflow.python.training import training_util
# from tensorflow.python.util import compat
# from tensorflow.python.util import nest


# def _eager_safe_var_handle_op(*args, **kwargs):
#   # When running in eager mode the `shared_name` should be set to the
#   # `anonymous_name` to avoid spurious sharing issues. The runtime generates a
#   # unique name on our behalf when the reserved `anonymous_name` is used as the
#   # `shared_name`.
#   if context.executing_eagerly() and "shared_name" not in kwargs:
#     kwargs["shared_name"] = context.anonymous_name()
#   return resource_variable_ops.var_handle_op(*args, **kwargs)


# class ResourceVariableOpsTest(test_util.TensorFlowTestCase,
#                               parameterized.TestCase):

#   def tearDown(self):
#     gc.collect()
#     # This will only contain uncollectable garbage, i.e. reference cycles
#     # involving objects with __del__ defined.
#     self.assertEmpty(gc.garbage)
#     super(ResourceVariableOpsTest, self).tearDown()

#   def testScatterUpdate(self):
#     with context.eager_mode():
#       v = resource_variable_ops.ResourceVariable([1.0, 2.0], name="update")
#       state_ops.scatter_update(v, [1], [3.0])
#       self.assertAllEqual([1.0, 3.0], v.numpy())

# if __name__ == "__main__":
#       test.main()
