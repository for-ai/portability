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
from tensorflow.python.summary.writer import writer_cache
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import monitored_session
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from ..utils.timer_wrapper import tensorflow_op_timer
from ..tensorflow_test import device_context
import tensorflow as tf

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


class CheckpointSaverHookTest(test.TestCase):

    def setUp(self):
        self.model_dir = tempfile.mkdtemp()
        self.graph = ops.Graph()
        with self.graph.as_default():
            self.scaffold = monitored_session.Scaffold()
            self.global_step = training_util.get_or_create_global_step()
            self.train_op = training_util._increment_global_step(1)

    def tearDown(self):
        shutil.rmtree(self.model_dir, ignore_errors=True)

    def test_saves_when_saver_and_scaffold_both_missing(self):
        with self.graph.as_default():
            with device_context():
                with tensorflow_op_timer():
                    hook = basic_session_run_hooks.CheckpointSaverHook(
                    self.model_dir, save_steps=1)
                hook.begin()
                self.scaffold.finalize()
                with session_lib.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess: 
                    sess.run(self.scaffold.init_op)
                    mon_sess = monitored_session._HookedSession(sess, [hook])
                    mon_sess.run(self.train_op)
                    self.assertEqual(1,
                                    checkpoint_utils.load_variable(self.model_dir,
                                                                    self.global_step.name))

    def test_raise_when_saver_and_scaffold_both_present(self):
        with self.assertRaises(ValueError):
            basic_session_run_hooks.CheckpointSaverHook(
                self.model_dir, saver=self.scaffold.saver, scaffold=self.scaffold)

    @test_util.run_deprecated_v1
    def test_raise_in_both_secs_and_steps(self):
        with self.assertRaises(ValueError):
            basic_session_run_hooks.CheckpointSaverHook(
                self.model_dir, save_secs=10, save_steps=20)

    @test_util.run_deprecated_v1
    def test_raise_in_none_secs_and_steps(self):
        with self.assertRaises(ValueError):
            basic_session_run_hooks.CheckpointSaverHook(self.model_dir)

    def test_save_secs_saves_in_first_step(self):
        with self.graph.as_default():
            with device_context():
                with tensorflow_op_timer():
                    hook = basic_session_run_hooks.CheckpointSaverHook(
                    self.model_dir, save_secs=2, scaffold=self.scaffold)
                hook.begin()
                self.scaffold.finalize()
                with session_lib.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
                    sess.run(self.scaffold.init_op)
                    mon_sess = monitored_session._HookedSession(sess, [hook])
                    mon_sess.run(self.train_op)
                    self.assertEqual(1,
                                    checkpoint_utils.load_variable(self.model_dir,
                                                                    self.global_step.name))

    def test_save_secs_calls_listeners_at_begin_and_end(self):
        with self.graph.as_default():
            with device_context():
                listener = MockCheckpointSaverListener()
                with tensorflow_op_timer():
                    hook = basic_session_run_hooks.CheckpointSaverHook(
                    self.model_dir,
                    save_secs=2,
                    scaffold=self.scaffold,
                    listeners=[listener])
                hook.begin()
                self.scaffold.finalize()
                with session_lib.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
                    sess.run(self.scaffold.init_op)
                    mon_sess = monitored_session._HookedSession(sess, [hook])
                    mon_sess.run(self.train_op)  # hook runs here
                    # hook won't run here, so it does at end
                    mon_sess.run(self.train_op)
                    hook.end(sess)  # hook runs here
                self.assertEqual({
                    'begin': 1,
                    'before_save': 2,
                    'after_save': 2,
                    'end': 1
                }, listener.get_counts())

    def test_listener_with_monitored_session(self):
        with ops.Graph().as_default():
            with device_context():
                scaffold = monitored_session.Scaffold()
                global_step = training_util.get_or_create_global_step()
                train_op = training_util._increment_global_step(1)
                listener = MockCheckpointSaverListener()
                with tensorflow_op_timer():
                    hook = basic_session_run_hooks.CheckpointSaverHook(
                    self.model_dir,
                    save_steps=1,
                    scaffold=scaffold,
                    listeners=[listener])
                with monitored_session.SingularMonitoredSession(
                        config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True),
                        hooks=[hook],
                        scaffold=scaffold,
                        checkpoint_dir=self.model_dir) as sess:
                    sess.run(train_op)
                    sess.run(train_op)
                    global_step_val = sess.raw_session().run(global_step)
                listener_counts = listener.get_counts()
            self.assertEqual(2, global_step_val)
            self.assertEqual({
                'begin': 1,
                'before_save': 3,
                'after_save': 3,
                'end': 1
            }, listener_counts)

    def test_listener_stops_training_in_after_save(self):
        with ops.Graph().as_default():
            with device_context():
                scaffold = monitored_session.Scaffold()
                training_util.get_or_create_global_step()
                train_op = training_util._increment_global_step(1)
                listener = MockCheckpointSaverListener()
                with tensorflow_op_timer():
                    hook = basic_session_run_hooks.CheckpointSaverHook(
                    self.model_dir, save_steps=1, scaffold=scaffold, listeners=[listener])
                with monitored_session.SingularMonitoredSession(
                        config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True),
                        hooks=[hook], scaffold=scaffold,
                        checkpoint_dir=self.model_dir) as sess:
                    sess.run(train_op)
                    self.assertFalse(sess.should_stop())
                    sess.run(train_op)
                    self.assertFalse(sess.should_stop())
                    listener.ask_for_stop = True
                    sess.run(train_op)
                    self.assertTrue(sess.should_stop())

    def test_listener_with_default_saver(self):
        with ops.Graph().as_default():
            with device_context():
                global_step = training_util.get_or_create_global_step()
                train_op = training_util._increment_global_step(1)
                listener = MockCheckpointSaverListener()
                with tensorflow_op_timer():
                    hook = basic_session_run_hooks.CheckpointSaverHook(
                    self.model_dir,
                    save_steps=1,
                    listeners=[listener])
                with monitored_session.SingularMonitoredSession(
                        config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True),
                        hooks=[hook],
                        checkpoint_dir=self.model_dir) as sess:
                    sess.run(train_op)
                    sess.run(train_op)
                    global_step_val = sess.raw_session().run(global_step)
                listener_counts = listener.get_counts()
            self.assertEqual(2, global_step_val)
            self.assertEqual({
                'begin': 1,
                'before_save': 3,
                'after_save': 3,
                'end': 1
            }, listener_counts)

            with ops.Graph().as_default():
                global_step = training_util.get_or_create_global_step()
                with monitored_session.SingularMonitoredSession(
                        config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True),
                        checkpoint_dir=self.model_dir) as sess2:
                    global_step_saved_val = sess2.run(global_step)
            self.assertEqual(2, global_step_saved_val)

    def test_two_listeners_with_default_saver(self):
        with ops.Graph().as_default():
            with device_context():
                global_step = training_util.get_or_create_global_step()
                train_op = training_util._increment_global_step(1)
                listener1 = MockCheckpointSaverListener()
                listener2 = MockCheckpointSaverListener()
                with tensorflow_op_timer():
                    hook = basic_session_run_hooks.CheckpointSaverHook(
                    self.model_dir,
                    save_steps=1,
                    listeners=[listener1, listener2])
                with monitored_session.SingularMonitoredSession(
                        config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True),
                        hooks=[hook],
                        checkpoint_dir=self.model_dir) as sess:
                    sess.run(train_op)
                    sess.run(train_op)
                    global_step_val = sess.raw_session().run(global_step)
                listener1_counts = listener1.get_counts()
                listener2_counts = listener2.get_counts()
        self.assertEqual(2, global_step_val)
        self.assertEqual({
            'begin': 1,
            'before_save': 3,
            'after_save': 3,
            'end': 1
        }, listener1_counts)
        self.assertEqual(listener1_counts, listener2_counts)

        with ops.Graph().as_default():
            global_step = training_util.get_or_create_global_step()
            with monitored_session.SingularMonitoredSession(
                    config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True),
                    checkpoint_dir=self.model_dir) as sess2:
                global_step_saved_val = sess2.run(global_step)
        self.assertEqual(2, global_step_saved_val)

    @test.mock.patch.object(time, 'time')
    def test_save_secs_saves_periodically(self, mock_time):
        with self.graph.as_default():
            with device_context():
                mock_time.return_value = MOCK_START_TIME
                with tensorflow_op_timer():
                    hook = basic_session_run_hooks.CheckpointSaverHook(
                    self.model_dir, save_secs=2, scaffold=self.scaffold)
                hook.begin()
                self.scaffold.finalize()

                with session_lib.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
                    sess.run(self.scaffold.init_op)
                    mon_sess = monitored_session._HookedSession(sess, [hook])

                    mock_time.return_value = MOCK_START_TIME
                    mon_sess.run(self.train_op)  # Saved.

                    mock_time.return_value = MOCK_START_TIME + 0.5
                    mon_sess.run(self.train_op)  # Not saved.

                    self.assertEqual(1,
                                    checkpoint_utils.load_variable(self.model_dir,
                                                                    self.global_step.name))

                    # Simulate 2.5 seconds of sleep.
                    mock_time.return_value = MOCK_START_TIME + 2.5
                    mon_sess.run(self.train_op)  # Saved.

                    mock_time.return_value = MOCK_START_TIME + 2.6
                    mon_sess.run(self.train_op)  # Not saved.

                    mock_time.return_value = MOCK_START_TIME + 2.7
                    mon_sess.run(self.train_op)  # Not saved.

                    self.assertEqual(3,
                                    checkpoint_utils.load_variable(self.model_dir,
                                                                    self.global_step.name))

                    # Simulate 7.5 more seconds of sleep (10 seconds from start.
                    mock_time.return_value = MOCK_START_TIME + 10
                    mon_sess.run(self.train_op)  # Saved.
                    self.assertEqual(6,
                                    checkpoint_utils.load_variable(self.model_dir,
                                                                    self.global_step.name))

    @test.mock.patch.object(time, 'time')
    def test_save_secs_calls_listeners_periodically(self, mock_time):
        with self.graph.as_default():
            with device_context():
                mock_time.return_value = MOCK_START_TIME
                listener = MockCheckpointSaverListener()
                with tensorflow_op_timer():
                    hook = basic_session_run_hooks.CheckpointSaverHook(
                    self.model_dir,
                    save_secs=2,
                    scaffold=self.scaffold,
                    listeners=[listener])
                hook.begin()
                self.scaffold.finalize()
                with session_lib.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
                    sess.run(self.scaffold.init_op)
                    mon_sess = monitored_session._HookedSession(sess, [hook])

                    mock_time.return_value = MOCK_START_TIME + 0.5
                    mon_sess.run(self.train_op)  # hook runs here

                    mock_time.return_value = MOCK_START_TIME + 0.5
                    mon_sess.run(self.train_op)

                    mock_time.return_value = MOCK_START_TIME + 3.0
                    mon_sess.run(self.train_op)  # hook runs here

                    mock_time.return_value = MOCK_START_TIME + 3.5
                    mon_sess.run(self.train_op)

                    mock_time.return_value = MOCK_START_TIME + 4.0
                    mon_sess.run(self.train_op)

                    mock_time.return_value = MOCK_START_TIME + 6.5
                    mon_sess.run(self.train_op)  # hook runs here

                    mock_time.return_value = MOCK_START_TIME + 7.0
                    # hook won't run here, so it does at end
                    mon_sess.run(self.train_op)

                    mock_time.return_value = MOCK_START_TIME + 7.5
                    hook.end(sess)  # hook runs here
                self.assertEqual({
                    'begin': 1,
                    'before_save': 4,
                    'after_save': 4,
                    'end': 1
                }, listener.get_counts())

    def test_save_steps_saves_in_first_step(self):
        with self.graph.as_default():
            with device_context():
                with tensorflow_op_timer():
                    hook = basic_session_run_hooks.CheckpointSaverHook(
                    self.model_dir, save_steps=2, scaffold=self.scaffold)
                hook.begin()
                self.scaffold.finalize()
                with session_lib.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
                    sess.run(self.scaffold.init_op)
                    mon_sess = monitored_session._HookedSession(sess, [hook])
                    mon_sess.run(self.train_op)
                    self.assertEqual(1,
                                    checkpoint_utils.load_variable(self.model_dir,
                                                                    self.global_step.name))

    def test_save_steps_saves_periodically(self):
        with self.graph.as_default():
            with device_context():
                with tensorflow_op_timer():
                    hook = basic_session_run_hooks.CheckpointSaverHook(
                    self.model_dir, save_steps=2, scaffold=self.scaffold)
                hook.begin()
                self.scaffold.finalize()
                with session_lib.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
                    sess.run(self.scaffold.init_op)
                    mon_sess = monitored_session._HookedSession(sess, [hook])
                    mon_sess.run(self.train_op)
                    mon_sess.run(self.train_op)
                    # Not saved
                    self.assertEqual(1,
                                    checkpoint_utils.load_variable(self.model_dir,
                                                                    self.global_step.name))
                    mon_sess.run(self.train_op)
                    # saved
                    self.assertEqual(3,
                                    checkpoint_utils.load_variable(self.model_dir,
                                                                    self.global_step.name))
                    mon_sess.run(self.train_op)
                    # Not saved
                    self.assertEqual(3,
                                    checkpoint_utils.load_variable(self.model_dir,
                                                                    self.global_step.name))
                    mon_sess.run(self.train_op)
                    # saved
                    self.assertEqual(5,
                                    checkpoint_utils.load_variable(self.model_dir,
                                                                    self.global_step.name))

    def test_save_saves_at_end(self):
        with self.graph.as_default():
            with device_context():
                with tensorflow_op_timer():
                    hook = basic_session_run_hooks.CheckpointSaverHook(
                    self.model_dir, save_secs=2, scaffold=self.scaffold)
                hook.begin()
                self.scaffold.finalize()
                with session_lib.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
                    sess.run(self.scaffold.init_op)
                    mon_sess = monitored_session._HookedSession(sess, [hook])
                    mon_sess.run(self.train_op)
                    mon_sess.run(self.train_op)
                    hook.end(sess)
                    self.assertEqual(2,
                                    checkpoint_utils.load_variable(self.model_dir,
                                                                    self.global_step.name))

    def test_save_checkpoint_before_first_train_step(self):
        with self.graph.as_default():
            with device_context():
                with tensorflow_op_timer():
                    hook = basic_session_run_hooks.CheckpointSaverHook(
                    self.model_dir, save_steps=2, scaffold=self.scaffold)
                hook.begin()
                self.scaffold.finalize()
                with session_lib.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
                    mon_sess = monitored_session._HookedSession(sess, [hook])
                    sess.run(self.scaffold.init_op)
                    hook.after_create_session(sess, None)
                    # Verifies that checkpoint is saved at step 0.
                    self.assertEqual(0,
                                    checkpoint_utils.load_variable(self.model_dir,
                                                                    self.global_step.name))
                    # Verifies that no checkpoint is saved after one training step.
                    mon_sess.run(self.train_op)
                    self.assertEqual(0,
                                    checkpoint_utils.load_variable(self.model_dir,
                                                                    self.global_step.name))
                    # Verifies that checkpoint is saved after save_steps.
                    mon_sess.run(self.train_op)
                    self.assertEqual(2,
                                    checkpoint_utils.load_variable(self.model_dir,
                                                                    self.global_step.name))

    def test_save_graph_def(self):
        with self.graph.as_default():
            with device_context():
                with tensorflow_op_timer():
                    hook = basic_session_run_hooks.CheckpointSaverHook(
                    self.model_dir, save_steps=1, scaffold=self.scaffold,
                    save_graph_def=True)
                hook.begin()
                self.scaffold.finalize()
                with session_lib.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
                    sess.run(self.scaffold.init_op)
                    mon_sess = monitored_session._HookedSession(sess, [hook])
                    sess.run(self.scaffold.init_op)
                    hook.after_create_session(sess, None)

                    self.assertIn('graph.pbtxt', os.listdir(self.model_dir))
                    # Should have a single .meta file for step 0
                    self.assertLen(gfile.Glob(
                        os.path.join(self.model_dir, '*.meta')), 1)

                    mon_sess.run(self.train_op)
                    self.assertLen(gfile.Glob(
                        os.path.join(self.model_dir, '*.meta')), 2)

    def test_save_graph_def_false(self):
        with self.graph.as_default():
            with device_context():
                with tensorflow_op_timer():
                    hook = basic_session_run_hooks.CheckpointSaverHook(
                    self.model_dir, save_steps=1, scaffold=self.scaffold,
                    save_graph_def=False)
                hook.begin()
                self.scaffold.finalize()
                with session_lib.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
                    sess.run(self.scaffold.init_op)
                    mon_sess = monitored_session._HookedSession(sess, [hook])
                    sess.run(self.scaffold.init_op)
                    hook.after_create_session(sess, None)

                    self.assertNotIn('graph.pbtxt', os.listdir(self.model_dir))
                    # Should have a single .meta file for step 0
                    self.assertEmpty(gfile.Glob(
                        os.path.join(self.model_dir, '*.meta')))

                    mon_sess.run(self.train_op)
                    self.assertEmpty(gfile.Glob(
                        os.path.join(self.model_dir, '*.meta')))


if __name__ == '__main__':
    test.main()
