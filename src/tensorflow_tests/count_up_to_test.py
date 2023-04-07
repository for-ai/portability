# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.resource_variable_ops."""
import gc

from absl.testing import parameterized

from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import test


def _eager_safe_var_handle_op(*args, **kwargs):
    # When running in eager mode the `shared_name` should be set to the
    # `anonymous_name` to avoid spurious sharing issues. The runtime generates a
    # unique name on our behalf when the reserved `anonymous_name` is used as the
    # `shared_name`.
    if context.executing_eagerly() and "shared_name" not in kwargs:
        kwargs["shared_name"] = context.anonymous_name()
    return resource_variable_ops.var_handle_op(*args, **kwargs)


@test_util.with_eager_op_as_function
@test_util.with_control_flow_v2
class ResourceVariableOpsTest(test_util.TensorFlowTestCase,
                              parameterized.TestCase):

    def tearDown(self):
        gc.collect()
        # This will only contain uncollectable garbage, i.e. reference cycles
        # involving objects with __del__ defined.
        self.assertEmpty(gc.garbage)
        super(ResourceVariableOpsTest, self).tearDown()

    def testCountUpToFunction(self):
        with context.eager_mode():
            v = resource_variable_ops.ResourceVariable(0, name="upto")
            self.assertAllEqual(state_ops.count_up_to(v, 1), 0)
            with self.assertRaises(errors.OutOfRangeError):
                state_ops.count_up_to(v, 1)


if __name__ == "__main__":
    test.main()
