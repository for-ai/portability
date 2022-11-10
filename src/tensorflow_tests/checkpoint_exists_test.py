"""Tests for tensorflow.python.training.saver.py."""

import contextlib
import os
import pathlib
import shutil
import tempfile

from google.protobuf import text_format

from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.checkpoint import checkpoint as util
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as saver_module
from tensorflow.python.training.checkpoint_state_pb2 import CheckpointState


class SaverUtilsTest(test.TestCase):
    def setUp(self):
        self._base_dir = os.path.join(self.get_temp_dir(), "saver_utils_test")
        gfile.MakeDirs(self._base_dir)

    def tearDown(self):
        gfile.DeleteRecursively(self._base_dir)

        def testCheckpointExists(self):
            for sharded in (False, True):
                for version in (saver_pb2.SaverDef.V2, saver_pb2.SaverDef.V1):
                    with self.session(graph=ops_lib.Graph()) as sess:
                        unused_v = variables.Variable(1.0, name="v")
                        self.evaluate(variables.global_variables_initializer())
                        saver = saver_module.Saver(
                            sharded=sharded, write_version=version)

                        path = os.path.join(
                            self._base_dir, "%s-%s" % (sharded, version))
                        self.assertFalse(
                            checkpoint_management.checkpoint_exists(path))  # Not saved yet.

                        ckpt_prefix = saver.save(sess, path)
                        self.assertTrue(
                            checkpoint_management.checkpoint_exists(ckpt_prefix))

                        ckpt_prefix = checkpoint_management.latest_checkpoint(
                            self._base_dir)
                        self.assertTrue(
                            checkpoint_management.checkpoint_exists(ckpt_prefix))


if __name__ == "__main__":
    test.main()
