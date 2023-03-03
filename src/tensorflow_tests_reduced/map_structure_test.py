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
"""Tests for utilities working with arbitrarily nested structures."""

import collections
import numpy as np
from absl.testing import parameterized

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.util import nest
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test


class NestTest(test_base.DatasetTestBase, parameterized.TestCase):

    @combinations.generate(test_base.default_test_combinations())
    def testMapStructure(self):
        structure1 = (((1, 2), 3), 4, (5, 6))
        structure2 = (((7, 8), 9), 10, (11, 12))
        structure1_plus1 = nest.map_structure(lambda x: x + 1, structure1)
        nest.assert_same_structure(structure1, structure1_plus1)
        self.assertAllEqual(
            [2, 3, 4, 5, 6, 7],
            nest.flatten(structure1_plus1))
        structure1_plus_structure2 = nest.map_structure(
            lambda x, y: x + y, structure1, structure2)
        self.assertEqual(
            (((1 + 7, 2 + 8), 3 + 9), 4 + 10, (5 + 11, 6 + 12)),
            structure1_plus_structure2)

        self.assertEqual(3, nest.map_structure(lambda x: x - 1, 4))

        self.assertEqual(7, nest.map_structure(lambda x, y: x + y, 3, 4))

        with self.assertRaisesRegex(TypeError, "callable"):
            nest.map_structure("bad", structure1_plus1)

        with self.assertRaisesRegex(ValueError, "same nested structure"):
            nest.map_structure(lambda x, y: None, 3, (3,))

        with self.assertRaisesRegex(TypeError, "same sequence type"):
            nest.map_structure(lambda x, y: None, ((3, 4), 5), {
                               "a": (3, 4), "b": 5})

        with self.assertRaisesRegex(ValueError, "same nested structure"):
            nest.map_structure(lambda x, y: None, ((3, 4), 5), (3, (4, 5)))

        with self.assertRaisesRegex(ValueError, "same nested structure"):
            nest.map_structure(lambda x, y: None, ((3, 4), 5), (3, (4, 5)),
                               check_types=False)

        with self.assertRaisesRegex(ValueError, "Only valid keyword argument"):
            nest.map_structure(lambda x: None, structure1, foo="a")

        with self.assertRaisesRegex(ValueError, "Only valid keyword argument"):
            nest.map_structure(lambda x: None, structure1,
                               check_types=False, foo="a")


if __name__ == "__main__":
    test.main()
