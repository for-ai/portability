"""Tests for V2 summary ops from summary_ops_v2."""

import os
import unittest

from tensorflow.core.util import event_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import tf_record
from tensorflow.python.ops import summary_ops_v2 as summary_ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from ..utils.timer_wrapper import tensorflow_op_timer


def events_from_file(filepath):
    """Returns all events in a single event file.

    Args:
      filepath: Path to the event file.

    Returns:
      A list of all tf.Event protos in the event file.
    """
    records = list(tf_record.tf_record_iterator(filepath))
    result = []
    for r in records:
        event = event_pb2.Event()
        event.ParseFromString(r)
        result.append(event)
    return result


def events_from_logdir(logdir):
    """Returns all events in the single eventfile in logdir.

    Args:
      logdir: The directory in which the single event file is sought.

    Returns:
      A list of all tf.Event protos from the single event file.

    Raises:
      AssertionError: If logdir does not contain exactly one file.
    """
    assert gfile.Exists(logdir)
    files = gfile.ListDirectory(logdir)
    assert len(files) == 1, 'Found not exactly one file in logdir: %s' % files
    return events_from_file(os.path.join(logdir, files[0]))


class SummaryOpsCoreTest(test_util.TensorFlowTestCase):

    def testWriterFlush(self):
        logdir = self.get_temp_dir()
        def get_total(): return len(events_from_logdir(logdir))
        with context.eager_mode():
            writer = summary_ops.create_file_writer_v2(
                logdir, max_queue=1000, flush_millis=1000000)
            self.assertEqual(1, get_total())  # file_version Event
            with writer.as_default():
                summary_ops.write('tag', 1, step=0)
                self.assertEqual(1, get_total())
                timer = tensorflow_op_timer()
                with timer:
                  test = writer.flush()
                  timer.gen.send(test)
                self.assertEqual(2, get_total())
                summary_ops.write('tag', 1, step=0)
                self.assertEqual(2, get_total())
            # Exiting the "as_default()" should do an implicit flush
            self.assertEqual(3, get_total())


if __name__ == '__main__':
    test.main()
