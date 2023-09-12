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
"""Tests for Reader ops from io_ops."""

import collections
import gzip
import os
import shutil
import sys
import threading
import zlib

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import tf_record
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow.python.training import input as input_lib
from tensorflow.python.training import queue_runner_impl
from tensorflow.python.util import compat
from ..utils.timer_wrapper import tensorflow_op_timer
prefix_path = "tensorflow/core/lib"

# pylint: disable=invalid-name
TFRecordCompressionType = tf_record.TFRecordCompressionType
# pylint: enable=invalid-name

# Edgar Allan Poe's 'Eldorado'
_TEXT = b"""Gaily bedight,
    A gallant knight,
    In sunshine and in shadow,
    Had journeyed long,
    Singing a song,
    In search of Eldorado.

    But he grew old
    This knight so bold
    And o'er his heart a shadow
    Fell as he found
    No spot of ground
    That looked like Eldorado.

   And, as his strength
   Failed him at length,
   He met a pilgrim shadow
   'Shadow,' said he,
   'Where can it be
   This land of Eldorado?'

   'Over the Mountains
    Of the Moon'
    Down the Valley of the Shadow,
    Ride, boldly ride,'
    The shade replied,
    'If you seek for Eldorado!'
    """


class TFCompressionTestCase(test.TestCase):
    
  def setUp(self):
    super(TFCompressionTestCase, self).setUp()
    self._num_files = 2
    self._num_records = 7

  def _Record(self, f, r):
    return compat.as_bytes("Record %d of file %d" % (r, f))

  def _CreateFiles(self, options=None, prefix=""):
    filenames = []
    for i in range(self._num_files):
      name = prefix + "tfrecord.%d.txt" % i
      records = [self._Record(i, j) for j in range(self._num_records)]
      fn = self._WriteRecordsToFile(records, name, options)
      filenames.append(fn)
    return filenames

  def _WriteRecordsToFile(self, records, name="tfrecord", options=None):
    fn = os.path.join(self.get_temp_dir(), name)
    with tf_record.TFRecordWriter(fn, options=options) as writer:
      for r in records:
        writer.write(r)
    return fn

  def _ZlibCompressFile(self, infile, name="tfrecord.z"):
    # zlib compress the file and write compressed contents to file.
    with open(infile, "rb") as f:
      cdata = zlib.compress(f.read())

    zfn = os.path.join(self.get_temp_dir(), name)
    with open(zfn, "wb") as f:
      f.write(cdata)
    return zfn

  def _GzipCompressFile(self, infile, name="tfrecord.gz"):
    # gzip compress the file and write compressed contents to file.
    with open(infile, "rb") as f:
      cdata = f.read()

    gzfn = os.path.join(self.get_temp_dir(), name)
    with gzip.GzipFile(gzfn, "wb") as f:
      f.write(cdata)
    return gzfn

  def _ZlibDecompressFile(self, infile, name="tfrecord"):
    with open(infile, "rb") as f:
      cdata = zlib.decompress(f.read())
    fn = os.path.join(self.get_temp_dir(), name)
    with open(fn, "wb") as f:
      f.write(cdata)
    return fn

  def _GzipDecompressFile(self, infile, name="tfrecord"):
    with gzip.GzipFile(infile, "rb") as f:
      cdata = f.read()
    fn = os.path.join(self.get_temp_dir(), name)
    with open(fn, "wb") as f:
      f.write(cdata)
    return fn


class IdentityReaderTest(test.TestCase):

  def _ExpectRead(self, key, value, expected):
    k, v = self.evaluate([key, value])
    self.assertAllEqual(expected, k)
    self.assertAllEqual(expected, v)

  @test_util.run_deprecated_v1
  def testSerializeRestore(self):
    reader = io_ops.IdentityReader("test_reader")
    produced = reader.num_records_produced()
    queue = data_flow_ops.FIFOQueue(99, [dtypes.string], shapes=())
    self.evaluate(queue.enqueue_many([["X", "Y", "Z"]]))
    key, value = reader.read(queue)

    self._ExpectRead(key, value, b"X")
    self.assertAllEqual(1, self.evaluate(produced))
    state = self.evaluate(reader.serialize_state())

    self._ExpectRead(key, value, b"Y")
    self._ExpectRead(key, value, b"Z")
    self.assertAllEqual(3, self.evaluate(produced))

    self.evaluate(queue.enqueue_many([["Y", "Z"]]))
    self.evaluate(queue.close())
    timer = tensorflow_op_timer()
    with timer:
      test = reader.restore_state(state)
      timer.gen.send(test)
    self.evaluate(reader.restore_state(state))
    self.assertAllEqual(1, self.evaluate(produced))
    self._ExpectRead(key, value, b"Y")
    self._ExpectRead(key, value, b"Z")
    with self.assertRaisesOpError("is closed and has insufficient elements "
                                  "\\(requested 1, current size 0\\)"):
      self.evaluate([key, value])
    self.assertAllEqual(3, self.evaluate(produced))

    self.assertEqual(bytes, type(state))

    with self.assertRaises(ValueError):
      reader.restore_state([])

    with self.assertRaises(ValueError):
      reader.restore_state([state, state])

    with self.assertRaisesOpError(
        "Could not parse state for IdentityReader 'test_reader'"):
      self.evaluate(reader.restore_state(state[1:]))

    with self.assertRaisesOpError(
        "Could not parse state for IdentityReader 'test_reader'"):
      self.evaluate(reader.restore_state(state[:-1]))

    with self.assertRaisesOpError(
        "Could not parse state for IdentityReader 'test_reader'"):
      self.evaluate(reader.restore_state(state + b"ExtraJunk"))

    with self.assertRaisesOpError(
        "Could not parse state for IdentityReader 'test_reader'"):
      self.evaluate(reader.restore_state(b"PREFIX" + state))

    with self.assertRaisesOpError(
        "Could not parse state for IdentityReader 'test_reader'"):
      self.evaluate(reader.restore_state(b"BOGUS" + state[5:]))



if __name__ == "__main__":
  test.main()
