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
"""Tests for lookup ops."""
import os
import tempfile
import unittest

from absl.testing import parameterized
import numpy as np

from tensorflow.python import tf2
from tensorflow.python.checkpoint import checkpoint as trackable
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.checkpoint import util as checkpoint_util
from tensorflow.python.client import session
from tensorflow.python.data.experimental.ops import counter
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load as saved_model_load
from tensorflow.python.saved_model import save as saved_model_save
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import autotrackable
from tensorflow.python.training import saver
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat


class BaseLookupTableTest(test.TestCase):

    def getHashTable(self):
        if tf2.enabled():
            return lookup_ops.StaticHashTable
        else:
            return lookup_ops.StaticHashTableV1

    def getVocabularyTable(self):
        if tf2.enabled():
            return lookup_ops.StaticVocabularyTable
        else:
            return lookup_ops.StaticVocabularyTableV1

    def initialize_table(self, table):
        if not tf2.enabled():
            self.evaluate(table.initializer)


@parameterized.named_parameters(
    (f"_{is_anonymous}", is_anonymous) for is_anonymous in [False, True])
class InitializeTableFromFileOpTest(BaseLookupTableTest):

    def _createVocabFile(self, basename, values=("brain", "salad", "surgery")):
        vocabulary_file = os.path.join(self.get_temp_dir(), basename)
        with open(vocabulary_file, "w") as f:
            f.write("\n".join(values) + "\n")
        return vocabulary_file

    def testInitializeStringTable(self, is_anonymous):
        if is_anonymous and not tf2.enabled():
            self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
        vocabulary_file = self._createVocabFile("one_column_1.txt")
        default_value = -1
        init = lookup_ops.TextFileInitializer(
            vocabulary_file, dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
            dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER)
        self.assertIn("one_column_1.txt_-2_-1", init._shared_name)
        table = self.getHashTable()(
            init, default_value, experimental_is_anonymous=is_anonymous)
        self.initialize_table(table)

        output = table.lookup(constant_op.constant(["brain", "salad", "tank"]))

        result = self.evaluate(output)
        self.assertAllEqual([0, 1, -1], result)

    def testInitializeInt64Table(self, is_anonymous):
        if is_anonymous and not tf2.enabled():
            self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
        vocabulary_file = self._createVocabFile(
            "one_column_int64.txt", values=("42", "1", "-1000"))

        with self.cached_session():
            default_value = -1
            init = lookup_ops.TextFileInitializer(
                vocabulary_file, dtypes.int64, lookup_ops.TextFileIndex.WHOLE_LINE,
                dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER)
            self.assertIn("one_column_int64.txt_-2_-1", init._shared_name)
            table = self.getHashTable()(
                init, default_value, experimental_is_anonymous=is_anonymous)
            self.initialize_table(table)

            output = table.lookup(
                constant_op.constant((42, 1, 11), dtype=dtypes.int64))

            result = self.evaluate(output)
            self.assertAllEqual([0, 1, -1], result)

    def testInitializeIndexTable(self, is_anonymous):
        if is_anonymous and not tf2.enabled():
            self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
        vocabulary_file = self._createVocabFile("one_column_2.txt")

        with self.cached_session():
            default_value = "UNK"
            key_index = lookup_ops.TextFileIndex.LINE_NUMBER
            value_index = lookup_ops.TextFileIndex.WHOLE_LINE
            init = lookup_ops.TextFileInitializer(
                vocabulary_file, dtypes.int64, key_index, dtypes.string, value_index)
            self.assertIn("one_column_2.txt_-1_-2", init._shared_name)
            table = self.getHashTable()(
                init, default_value, experimental_is_anonymous=is_anonymous)
            self.initialize_table(table)

            input_values = constant_op.constant([0, 1, 2, 3], dtypes.int64)
            output = table.lookup(input_values)

            result = self.evaluate(output)
            self.assertAllEqual(
                [b"brain", b"salad", b"surgery", b"UNK"], result)

    def testMultiColumn(self, is_anonymous):
        if is_anonymous and not tf2.enabled():
            self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
        vocabulary_file = os.path.join(
            self.get_temp_dir(), "three_columns.txt")
        with open(vocabulary_file, "w") as f:
            f.write(
                "\n".join(["0\tbrain\t1", "1\tsalad\t5", "2\tsurgery\t6"]) + "\n")

        with self.cached_session():
            default_value = -1
            key_index = 1
            value_index = 2

            init = lookup_ops.TextFileInitializer(
                vocabulary_file, dtypes.string, key_index, dtypes.int64, value_index)
            self.assertIn("three_columns.txt_1_2", init._shared_name)
            table = self.getHashTable()(
                init, default_value, experimental_is_anonymous=is_anonymous)
            self.initialize_table(table)

            input_string = constant_op.constant(["brain", "salad", "surgery"])
            output = table.lookup(input_string)

            result = self.evaluate(output)
            self.assertAllEqual([1, 5, 6], result)

    def testInvalidDataTypeInMultiColumn(self, is_anonymous):
        vocabulary_file = os.path.join(
            self.get_temp_dir(), "three_columns.txt")
        with open(vocabulary_file, "w") as f:
            f.write(
                "\n".join(["0\tbrain\t1", "1\tsalad\t5", "2\tsurgery\t6"]) + "\n")

        with self.cached_session():
            default_value = -1
            key_index = 2
            value_index = 1
            init = lookup_ops.TextFileInitializer(
                vocabulary_file, dtypes.string, key_index, dtypes.int64, value_index)
            self.assertIn("three_columns.txt_2_1", init._shared_name)
            with self.assertRaisesOpError("is not a valid"):
                table = self.getHashTable()(
                    init, default_value, experimental_is_anonymous=is_anonymous)
                self.initialize_table(table)

    def testInvalidDataType(self, is_anonymous):
        vocabulary_file = self._createVocabFile("one_column_3.txt")

        with self.cached_session():
            default_value = "UNK"
            key_index = lookup_ops.TextFileIndex.WHOLE_LINE
            value_index = lookup_ops.TextFileIndex.LINE_NUMBER

            with self.assertRaises(ValueError):
                init = lookup_ops.TextFileInitializer(vocabulary_file, dtypes.int64,
                                                      key_index, dtypes.string,
                                                      value_index)
                self.assertIn("one_column_3.txt_-2_-1", init._shared_name)
                self.getHashTable()(
                    init, default_value, experimental_is_anonymous=is_anonymous)

    def testInvalidIndex(self, is_anonymous):
        vocabulary_file = self._createVocabFile("one_column_4.txt")
        with self.cached_session():
            default_value = -1
            key_index = 1  # second column of the line
            value_index = lookup_ops.TextFileIndex.LINE_NUMBER
            init = lookup_ops.TextFileInitializer(
                vocabulary_file, dtypes.string, key_index, dtypes.int64, value_index)
            self.assertIn("one_column_4.txt_1_-1", init._shared_name)

            with self.assertRaisesOpError("Invalid number of columns"):
                table = self.getHashTable()(
                    init, default_value, experimental_is_anonymous=is_anonymous)
                self.initialize_table(table)

    def testInitializeSameTableWithMultipleNodes(self, is_anonymous):
        if is_anonymous and not tf2.enabled():
            self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
        vocabulary_file = self._createVocabFile("one_column_5.txt")

        with self.cached_session():
            default_value = -1
            init1 = lookup_ops.TextFileInitializer(
                vocabulary_file, dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
                dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER)
            self.assertIn("one_column_5.txt_-2_-1", init1._shared_name)
            table1 = self.getHashTable()(
                init1, default_value, experimental_is_anonymous=is_anonymous)
            init2 = lookup_ops.TextFileInitializer(
                vocabulary_file, dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
                dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER)
            self.assertIn("one_column_5.txt_-2_-1", init2._shared_name)
            table2 = self.getHashTable()(
                init2, default_value, experimental_is_anonymous=is_anonymous)
            init3 = lookup_ops.TextFileInitializer(
                vocabulary_file, dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
                dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER)
            self.assertIn("one_column_5.txt_-2_-1", init3._shared_name)
            table3 = self.getHashTable()(
                init3, default_value, experimental_is_anonymous=is_anonymous)

            self.evaluate(lookup_ops.tables_initializer())

            input_string = constant_op.constant(["brain", "salad", "tank"])

            output1 = table1.lookup(input_string)
            output2 = table2.lookup(input_string)
            output3 = table3.lookup(input_string)

            out1, out2, out3 = self.evaluate([output1, output2, output3])
            self.assertAllEqual([0, 1, -1], out1)
            self.assertAllEqual([0, 1, -1], out2)
            self.assertAllEqual([0, 1, -1], out3)

    def testInitializeTableWithNoFilename(self, is_anonymous):
        with self.cached_session():
            default_value = -1
            with self.assertRaises(ValueError):
                self.getHashTable()(
                    lookup_ops.TextFileInitializer(
                        "", dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
                        dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER),
                    default_value,
                    experimental_is_anonymous=is_anonymous)

    def testInitializeWithVocabSize(self, is_anonymous):
        if is_anonymous and not tf2.enabled():
            self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
        with self.cached_session():
            default_value = -1
            vocab_size = 3
            vocabulary_file1 = self._createVocabFile("one_column6.txt")
            init1 = lookup_ops.TextFileInitializer(
                vocabulary_file1,
                dtypes.string,
                lookup_ops.TextFileIndex.WHOLE_LINE,
                dtypes.int64,
                lookup_ops.TextFileIndex.LINE_NUMBER,
                vocab_size=vocab_size)
            self.assertIn("one_column6.txt_3_-2_-1", init1._shared_name)
            table1 = self.getHashTable()(
                init1, default_value, experimental_is_anonymous=is_anonymous)

            # Initialize from file.
            self.initialize_table(table1)
            self.assertEqual(vocab_size, self.evaluate(table1.size()))

            vocabulary_file2 = self._createVocabFile("one_column7.txt")
            vocab_size = 5
            init2 = lookup_ops.TextFileInitializer(
                vocabulary_file2,
                dtypes.string,
                lookup_ops.TextFileIndex.WHOLE_LINE,
                dtypes.int64,
                lookup_ops.TextFileIndex.LINE_NUMBER,
                vocab_size=vocab_size)
            self.assertIn("one_column7.txt_5_-2_-1", init2._shared_name)
            with self.assertRaisesOpError("Invalid vocab_size"):
                table2 = self.getHashTable()(
                    init2, default_value, experimental_is_anonymous=is_anonymous)
                self.initialize_table(table2)

            vocab_size = 1
            vocabulary_file3 = self._createVocabFile("one_column3.txt")
            init3 = lookup_ops.TextFileInitializer(
                vocabulary_file3,
                dtypes.string,
                lookup_ops.TextFileIndex.WHOLE_LINE,
                dtypes.int64,
                lookup_ops.TextFileIndex.LINE_NUMBER,
                vocab_size=vocab_size)
            self.assertIn("one_column3.txt_1_-2_-1", init3._shared_name)
            table3 = self.getHashTable()(
                init3, default_value, experimental_is_anonymous=is_anonymous)

            # Smaller vocab size reads only vocab_size records.
            self.initialize_table(table3)
            self.assertEqual(vocab_size, self.evaluate(table3.size()))

    @test_util.run_v1_only("placeholder usage")
    def testFeedVocabularyName(self, is_anonymous):
        if is_anonymous and not tf2.enabled():
            self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
        vocabulary_file = self._createVocabFile("feed_vocabulary.txt")

        with self.cached_session():
            default_value = -1
            init = lookup_ops.TextFileInitializer(
                "old_file.txt", dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
                dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER)
            self.assertIn("old_file.txt_-2_-1", init._shared_name)
            table = self.getHashTable()(
                init, default_value, experimental_is_anonymous=is_anonymous)

            # Initialize with non existing file (old_file.txt) should fail.
            # TODO(yleon): Update message, which might change per FileSystem.
            with self.assertRaisesOpError("old_file.txt"):
                self.evaluate(table.initializer)

            # Initialize the model feeding the vocabulary file.
            filenames = ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS)
            table.initializer.run(feed_dict={filenames[0]: vocabulary_file})

            input_string = constant_op.constant(["brain", "salad", "tank"])
            output = table.lookup(input_string)

            result = self.evaluate(output)
            self.assertAllEqual([0, 1, -1], result)

    def testInvalidFilenames(self, is_anonymous):
        vocabulary_file = self._createVocabFile("filename_shape.txt")

        with self.cached_session():
            default_value = -1

            # Invalid data type
            other_type = constant_op.constant(1)
            with self.assertRaises(Exception) as cm:
                self.getHashTable()(
                    lookup_ops.TextFileInitializer(
                        other_type, dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
                        dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER),
                    default_value,
                    experimental_is_anonymous=is_anonymous)
            self.assertIsInstance(cm.exception, (ValueError, TypeError))

            # Non-scalar filename
            filenames = constant_op.constant(
                [vocabulary_file, vocabulary_file])
            if not context.executing_eagerly():
                with self.assertRaises(Exception) as cm:
                    self.getHashTable()(
                        lookup_ops.TextFileInitializer(
                            filenames, dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
                            dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER),
                        default_value,
                        experimental_is_anonymous=is_anonymous)
                self.assertIsInstance(cm.exception, (ValueError, TypeError))
            else:
                with self.assertRaises(errors_impl.InvalidArgumentError):
                    self.getHashTable()(
                        lookup_ops.TextFileInitializer(
                            filenames, dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
                            dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER),
                        default_value,
                        experimental_is_anonymous=is_anonymous)


if __name__ == "__main__":
    test.main()
