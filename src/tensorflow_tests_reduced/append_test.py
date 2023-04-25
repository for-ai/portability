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
"""Tests for lists module."""
import sys
from tensorflow.python.autograph.converters import lists
from tensorflow.python.autograph.lang import special_functions
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import list_ops
from tensorflow.python.autograph.core import converter_testing

from ..utils.timer_wrapper import tensorflow_op_timer
# from ..utils.tensorflow_contexts import PortabilityTestCase
import tensorflow as tf
from ..utils.timer_wrapper import tensorflow_op_timer

class ListTest(converter_testing.TestCase):
    def test_list_append(self):

        def f():
            l = special_functions.tensor_list([1])
            timer = tensorflow_op_timer()
            with timer:
                 l.append(2)
            with timer:
                 l.append(3)
            return l
        
        tr = self.transform(f, lists)
        tl = tr()
        r = list_ops.tensor_list_stack(tl, dtypes.int32)
        
        self.assertAllEqual(self.evaluate(r), [1, 2, 3])



if __name__ == '__main__':
    test.main()
