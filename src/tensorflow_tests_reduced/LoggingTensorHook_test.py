# pylint: disable=g-bad-file-header
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
"""Tests for basic_session_run_hooks."""

import os.path
import shutil
import tempfile
import time

from tensorflow.python.client import session as session_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
from tensorflow.python.summary import summary as summary_lib
# from tensorflow.python.summary.writer import fake_summary_writer
from tensorflow.python.summary.writer import writer_cache
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import monitored_session
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from ..utils.timer_wrapper import tensorflow_op_timer


# Provide a realistic start time for unit tests where we need to mock out
# calls to time.time().
MOCK_START_TIME = 1484695987.209386


class MockCheckpointSaverListener(
    basic_session_run_hooks.CheckpointSaverListener):

  def __init__(self):
    self.begin_count = 0
    self.before_save_count = 0
    self.after_save_count = 0
    self.end_count = 0
    self.ask_for_stop = False

  def begin(self):
    self.begin_count += 1

  def before_save(self, session, global_step):
    self.before_save_count += 1

  def after_save(self, session, global_step):
    self.after_save_count += 1
    if self.ask_for_stop:
      return True

  def end(self, session, global_step):
    self.end_count += 1

  def get_counts(self):
    return {
        'begin': self.begin_count,
        'before_save': self.before_save_count,
        'after_save': self.after_save_count,
        'end': self.end_count
    }



class LoggingTensorHookTest(test.TestCase):

  def setUp(self):
    # Mock out logging calls so we can verify whether correct tensors are being
    # monitored.
    self._actual_log = tf_logging.info
    self.logged_message = None

    def mock_log(*args, **kwargs):
      self.logged_message = args
      self._actual_log(*args, **kwargs)

    tf_logging.info = mock_log

  def tearDown(self):
    tf_logging.info = self._actual_log

  def test_illegal_args(self):
    with self.assertRaisesRegex(ValueError, 'nvalid every_n_iter'):
      basic_session_run_hooks.LoggingTensorHook(tensors=['t'], every_n_iter=0)
    with self.assertRaisesRegex(ValueError, 'nvalid every_n_iter'):
      basic_session_run_hooks.LoggingTensorHook(tensors=['t'], every_n_iter=-10)
    with self.assertRaisesRegex(ValueError, 'xactly one of'):
      basic_session_run_hooks.LoggingTensorHook(
          tensors=['t'], every_n_iter=5, every_n_secs=5)
    with self.assertRaisesRegex(ValueError, 'xactly one of'):
      basic_session_run_hooks.LoggingTensorHook(tensors=['t'])

  def test_print_at_end_only(self):
    with ops.Graph().as_default(), session_lib.Session() as sess:
      t = constant_op.constant(42.0, name='foo')
      train_op = constant_op.constant(3)
      timer = tensorflow_op_timer()
      with timer:
        hook = basic_session_run_hooks.LoggingTensorHook(
          tensors=[t.name], at_end=True)
        timer.gen.send(hook)
      hook.begin()
      mon_sess = monitored_session._HookedSession(sess, [hook])
      self.evaluate(variables_lib.global_variables_initializer())
      self.logged_message = ''
      for _ in range(3):
        mon_sess.run(train_op)
        # assertNotRegexpMatches is not supported by python 3.1 and later
        self.assertEqual(str(self.logged_message).find(t.name), -1)

      hook.end(sess)
      self.assertRegex(str(self.logged_message), t.name)

  def _validate_print_every_n_steps(self, sess, at_end):
    t = constant_op.constant(42.0, name='foo')

    train_op = constant_op.constant(3)
    timer = tensorflow_op_timer()
    with timer:
      hook = basic_session_run_hooks.LoggingTensorHook(
        tensors=[t.name], every_n_iter=10, at_end=at_end)
      timer.gen.send(hook)
    hook.begin()
    mon_sess = monitored_session._HookedSession(sess, [hook])
    self.evaluate(variables_lib.global_variables_initializer())
    mon_sess.run(train_op)
    self.assertRegex(str(self.logged_message), t.name)
    for _ in range(3):
      self.logged_message = ''
      for _ in range(9):
        mon_sess.run(train_op)
        # assertNotRegexpMatches is not supported by python 3.1 and later
        self.assertEqual(str(self.logged_message).find(t.name), -1)
      mon_sess.run(train_op)
      self.assertRegex(str(self.logged_message), t.name)

    # Add additional run to verify proper reset when called multiple times.
    self.logged_message = ''
    mon_sess.run(train_op)
    # assertNotRegexpMatches is not supported by python 3.1 and later
    self.assertEqual(str(self.logged_message).find(t.name), -1)

    self.logged_message = ''
    hook.end(sess)
    if at_end:
      self.assertRegex(str(self.logged_message), t.name)
    else:
      # assertNotRegexpMatches is not supported by python 3.1 and later
      self.assertEqual(str(self.logged_message).find(t.name), -1)

  def test_print_every_n_steps(self):
    with ops.Graph().as_default(), session_lib.Session() as sess:
      self._validate_print_every_n_steps(sess, at_end=False)
      # Verify proper reset.
      self._validate_print_every_n_steps(sess, at_end=False)

  def test_print_every_n_steps_and_end(self):
    with ops.Graph().as_default(), session_lib.Session() as sess:
      self._validate_print_every_n_steps(sess, at_end=True)
      # Verify proper reset.
      self._validate_print_every_n_steps(sess, at_end=True)

  def test_print_first_step(self):
    # if it runs every iteration, first iteration has None duration.
    with ops.Graph().as_default(), session_lib.Session() as sess:
      t = constant_op.constant(42.0, name='foo')
      train_op = constant_op.constant(3)
      timer = tensorflow_op_timer()
      with timer:
        hook = basic_session_run_hooks.LoggingTensorHook(
          tensors={'foo': t}, every_n_iter=1)
        timer.gen.send(hook)
      hook.begin()
      mon_sess = monitored_session._HookedSession(sess, [hook])
      self.evaluate(variables_lib.global_variables_initializer())
      mon_sess.run(train_op)
      self.assertRegex(str(self.logged_message), 'foo')
      # in first run, elapsed time is None.
      self.assertEqual(str(self.logged_message).find('sec'), -1)

  def _validate_print_every_n_secs(self, sess, at_end, mock_time):
    t = constant_op.constant(42.0, name='foo')
    train_op = constant_op.constant(3)
    timer = tensorflow_op_timer()
    with timer:
      hook = basic_session_run_hooks.LoggingTensorHook(
        tensors=[t.name], every_n_secs=1.0, at_end=at_end)
      timer.gen.send(hook)
    hook.begin()
    mon_sess = monitored_session._HookedSession(sess, [hook])
    self.evaluate(variables_lib.global_variables_initializer())

    mon_sess.run(train_op)
    self.assertRegex(str(self.logged_message), t.name)

    # assertNotRegexpMatches is not supported by python 3.1 and later
    self.logged_message = ''
    mon_sess.run(train_op)
    self.assertEqual(str(self.logged_message).find(t.name), -1)
    mock_time.return_value += 1.0

    self.logged_message = ''
    mon_sess.run(train_op)
    self.assertRegex(str(self.logged_message), t.name)

    self.logged_message = ''
    hook.end(sess)
    if at_end:
      self.assertRegex(str(self.logged_message), t.name)
    else:
      # assertNotRegexpMatches is not supported by python 3.1 and later
      self.assertEqual(str(self.logged_message).find(t.name), -1)

  @test.mock.patch.object(time, 'time')
  def test_print_every_n_secs(self, mock_time):
    with ops.Graph().as_default(), session_lib.Session() as sess:
      mock_time.return_value = MOCK_START_TIME
      self._validate_print_every_n_secs(sess, at_end=False, mock_time=mock_time)
      # Verify proper reset.
      self._validate_print_every_n_secs(sess, at_end=False, mock_time=mock_time)

  @test.mock.patch.object(time, 'time')
  def test_print_every_n_secs_and_end(self, mock_time):
    with ops.Graph().as_default(), session_lib.Session() as sess:
      mock_time.return_value = MOCK_START_TIME
      self._validate_print_every_n_secs(sess, at_end=True, mock_time=mock_time)
      # Verify proper reset.
      self._validate_print_every_n_secs(sess, at_end=True, mock_time=mock_time)

  def test_print_formatter(self):
    with ops.Graph().as_default(), session_lib.Session() as sess:
      t = constant_op.constant(42.0, name='foo')
      train_op = constant_op.constant(3)
      timer = tensorflow_op_timer()
      with timer:
        hook = basic_session_run_hooks.LoggingTensorHook(
          tensors=[t.name], every_n_iter=10,
          formatter=lambda items: 'qqq=%s' % items[t.name])
        timer.gen.send(hook)
      hook.begin()
      mon_sess = monitored_session._HookedSession(sess, [hook])
      self.evaluate(variables_lib.global_variables_initializer())
      mon_sess.run(train_op)
      self.assertEqual(self.logged_message[0], 'qqq=42.0')

if __name__ == '__main__':
  test.main()
