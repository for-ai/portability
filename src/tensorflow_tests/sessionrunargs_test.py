import collections
import glob
import os
import sys
import threading
import time
import traceback

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import debug_pb2
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.client import session as session_lib
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import distribute_coordinator
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import load as saved_model_load
from tensorflow.python.summary import summary
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import coordinator
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import summary_io
from tensorflow.python.training import training_util


def latest_summaries(base_dir):
  """Parse summary events from latest event file in base_dir."""
  file_paths = glob.glob(os.path.join(base_dir, 'events.*'))
  file_path = sorted(file_paths)[-1] if file_paths else None
  latest_events = summary_io.summary_iterator(file_path) if file_path else []
  return [e for e in latest_events if e.HasField('summary')]

class FakeHook(session_run_hook.SessionRunHook):
    
  def __init__(self):
    self.should_stop = False
    self.request = None
    self.call_counter = collections.Counter()
    self.last_run_context = None
    self.last_run_values = None

  def begin(self):
    self.call_counter['begin'] += 1

  def after_create_session(self, session, coord):  # pylint: disable=unused-argument
    self.call_counter['after_create_session'] += 1

  def before_run(self, run_context):
    self.call_counter['before_run'] += 1
    self.last_run_context = run_context
    return self.request

  def after_run(self, run_context, run_values):
    self.call_counter['after_run'] += 1
    self.last_run_values = run_values
    if self.should_stop:
      run_context.request_stop()

  def end(self, session):
    self.call_counter['end'] += 1


class CountingSessionCreator:
  """A creator that counts the number of created sessions."""

  def __init__(self, session):
    self._initial_session = session
    # We only have one session per test case. We can't re-create it, thus
    # it shouldn't be closed.
    self._initial_session.close = lambda *args: None
    self._create_session_calls = 0

  @property
  def number_of_sessions_created(self):
    return self._create_session_calls

  def create_session(self):
    self._create_session_calls += 1
    return self._initial_session


class HookedSessionTest(test.TestCase):
  """Tests of _HookedSession."""

  def testCallsHooksBeginEnd(self):
    with ops.Graph().as_default(), session_lib.Session() as sess:
      mock_hook = FakeHook()
      mock_hook2 = FakeHook()
      mon_sess = monitored_session._HookedSession(
          sess=sess, hooks=[mock_hook, mock_hook2])
      a_tensor = constant_op.constant([0], name='a_tensor')
      self.evaluate(variables.global_variables_initializer())
      mon_sess.run(a_tensor)

      for hook in [mock_hook, mock_hook2]:
        self.assertEqual(
            hook.last_run_values,
            session_run_hook.SessionRunValues(
                results=None,
                options=config_pb2.RunOptions(),
                run_metadata=config_pb2.RunMetadata()))
        self.assertEqual(hook.last_run_context.original_args,
                         session_run_hook.SessionRunArgs(a_tensor))
        self.assertEqual(hook.last_run_context.session, sess)
        self.assertEqual(hook.call_counter['begin'], 0)
        self.assertEqual(hook.call_counter['after_create_session'], 0)
        self.assertEqual(hook.call_counter['before_run'], 1)
        self.assertEqual(hook.call_counter['after_run'], 1)

  
  def testFetchesHookRequests(self):
    with ops.Graph().as_default(), session_lib.Session() as sess:
      mock_hook = FakeHook()
      mock_hook2 = FakeHook()
      mon_sess = monitored_session._HookedSession(
          sess=sess, hooks=[mock_hook, mock_hook2])
      a_tensor = constant_op.constant([0], name='a_tensor')
      another_tensor = constant_op.constant([5], name='another_tensor')
      third_tensor = constant_op.constant([10], name='third_tensor')
      mock_hook.request = session_run_hook.SessionRunArgs([another_tensor])
      mock_hook2.request = session_run_hook.SessionRunArgs([third_tensor])
      self.evaluate(variables.global_variables_initializer())

      output = mon_sess.run(fetches=a_tensor)
      self.assertEqual(output, [0])
      self.assertEqual(mock_hook.last_run_values.results, [5])
      self.assertEqual(mock_hook2.last_run_values.results, [10])

  def testOnlyHooksHaveFeeds(self):
    with ops.Graph().as_default(), session_lib.Session() as sess:
      mock_hook = FakeHook()
      mock_hook2 = FakeHook()
      mon_sess = monitored_session._HookedSession(
          sess=sess, hooks=[mock_hook, mock_hook2])
      a_tensor = constant_op.constant([0], name='a_tensor')
      b_tensor = constant_op.constant([0], name='b_tensor')
      add_tensor = a_tensor + b_tensor
      mock_hook.request = session_run_hook.SessionRunArgs(
          None, feed_dict={a_tensor: [5]})
      mock_hook2.request = session_run_hook.SessionRunArgs(
          None, feed_dict={b_tensor: [10]})
      self.evaluate(variables.global_variables_initializer())

      self.assertEqual(mon_sess.run(fetches=add_tensor), [15])

  def testBothHooksAndUserHaveFeeds(self):
    with ops.Graph().as_default(), session_lib.Session() as sess:
      mock_hook = FakeHook()
      mock_hook2 = FakeHook()
      mon_sess = monitored_session._HookedSession(
          sess=sess, hooks=[mock_hook, mock_hook2])
      a_tensor = constant_op.constant([0], name='a_tensor')
      b_tensor = constant_op.constant([0], name='b_tensor')
      c_tensor = constant_op.constant([0], name='c_tensor')
      add_tensor = a_tensor + b_tensor + c_tensor
      mock_hook.request = session_run_hook.SessionRunArgs(
          None, feed_dict={a_tensor: [5]})
      mock_hook2.request = session_run_hook.SessionRunArgs(
          None, feed_dict={b_tensor: [10]})
      self.evaluate(variables.global_variables_initializer())

      feed_dict = {c_tensor: [20]}
      self.assertEqual(
          mon_sess.run(fetches=add_tensor, feed_dict=feed_dict), [35])
      # User feed_dict should not be changed
      self.assertEqual(len(feed_dict), 1)

  def testHooksFeedConflicts(self):
    with ops.Graph().as_default(), session_lib.Session() as sess:
      mock_hook = FakeHook()
      mock_hook2 = FakeHook()
      mon_sess = monitored_session._HookedSession(
          sess=sess, hooks=[mock_hook, mock_hook2])
      a_tensor = constant_op.constant([0], name='a_tensor')
      b_tensor = constant_op.constant([0], name='b_tensor')
      add_tensor = a_tensor + b_tensor
      mock_hook.request = session_run_hook.SessionRunArgs(
          None, feed_dict={a_tensor: [5]})
      mock_hook2.request = session_run_hook.SessionRunArgs(
          None, feed_dict={a_tensor: [10]})
      self.evaluate(variables.global_variables_initializer())

      with self.assertRaisesRegex(RuntimeError, 'Same tensor is fed'):
        mon_sess.run(fetches=add_tensor)

  def testHooksAndUserFeedConflicts(self):
    with ops.Graph().as_default(), session_lib.Session() as sess:
      mock_hook = FakeHook()
      mock_hook2 = FakeHook()
      mon_sess = monitored_session._HookedSession(
          sess=sess, hooks=[mock_hook, mock_hook2])
      a_tensor = constant_op.constant([0], name='a_tensor')
      b_tensor = constant_op.constant([0], name='b_tensor')
      add_tensor = a_tensor + b_tensor
      mock_hook.request = session_run_hook.SessionRunArgs(
          None, feed_dict={a_tensor: [5]})
      mock_hook2.request = session_run_hook.SessionRunArgs(
          None, feed_dict={b_tensor: [10]})
      self.evaluate(variables.global_variables_initializer())

      with self.assertRaisesRegex(RuntimeError, 'Same tensor is fed'):
        mon_sess.run(fetches=add_tensor, feed_dict={b_tensor: [10]})


class RaiseOnceAtCountN(session_run_hook.SessionRunHook):
  """Hook that raises an Exception at step N."""

  def __init__(self, n, ex):
    self.n = n
    self.ex = ex
    self.raised = False

  def before_run(self, run_context):
    # Raise the first time we reach step N.
    self.n -= 1
    if 0 == self.n and not self.raised:
      self.raised = True
      raise self.ex
    return None


class RunOptionsMetadataHook(session_run_hook.SessionRunHook):
  """A hook that observes & optionally modifies RunOptions and RunMetadata."""

  def __init__(self, trace_level, timeout_in_ms, output_partition_graphs,
               debug_tensor_watch, report_tensor_allocations_upon_oom):
    self._trace_level = trace_level
    self._timeout_in_ms = timeout_in_ms
    self._output_partition_graphs = output_partition_graphs
    self._debug_tensor_watch = debug_tensor_watch
    self._report_tensor_allocations_upon_oom = (
        report_tensor_allocations_upon_oom)

    self.run_options_list = []
    self.run_metadata_list = []

  def before_run(self, run_context):
    options = config_pb2.RunOptions(
        trace_level=self._trace_level,
        timeout_in_ms=self._timeout_in_ms,
        output_partition_graphs=self._output_partition_graphs,
        report_tensor_allocations_upon_oom=self
        ._report_tensor_allocations_upon_oom)
    options.debug_options.debug_tensor_watch_opts.extend(
        [self._debug_tensor_watch])
    return session_run_hook.SessionRunArgs(None, None, options=options)

  def after_run(self, run_context, run_values):
    self.run_options_list.append(run_values.options)
    self.run_metadata_list.append(run_values.run_metadata)


class MonitoredSessionTest(test.TestCase):
  """MonitoredSession tests."""

  def test_step_fn_with_hooks(self):
    with ops.Graph().as_default():
      var = resource_variable_ops.ResourceVariable(0.0)

      # This test highlights the interaction of hooks with
      # `Monitoredsession.run_step_fn`.  The order of execution of operations
      # below is:
      #   0.  stage_0
      #   1.  stage_1_0 or stage_1_1 in an undefined order
      #   2.  stage_2

      stage_0 = state_ops.assign_add(var, 0.3)
      stage_1_0 = state_ops.assign_add(var, 0.7)
      # The order of `stage_1_0` and `stage_1_1` is undefined by
      # `MonitoredSession`, but we should be able to assert when both of them
      # are complete.  To obtain a consistent result of adding two different
      # constants to `var`, we rely on a control dependency and
      # `ResourceVariable`.  Otherwise, it is possible that one of the
      # additions overwrites the result of the other addition.
      with ops.control_dependencies([stage_1_0]):
        stage_1_1 = state_ops.assign_add(var, 0.5)
      stage_2 = state_ops.assign_add(var, 1.1)

      class Hook(session_run_hook.SessionRunHook):

        def __init__(self, testing):
          self._testing = testing

        def before_run(self, run_context):
          return session_run_hook.SessionRunArgs(fetches=stage_1_0)

        def after_run(self, run_context, run_values):
          self._testing.assertNear(0.3 + 0.5 + 0.7,
                                   run_context.session.run(var), 0.1)
          self._testing.assertNear(0.3 + 0.5 + 0.7 + 1.1,
                                   run_context.session.run(stage_2), 0.1)

      def step_fn(step_context):
        self.assertNear(0.3, step_context.session.run(stage_0), 0.1)
        return step_context.run_with_hooks(fetches=stage_1_1)

      with monitored_session.MonitoredSession(hooks=[Hook(self)]) as session:
        self.assertEqual(0.3 + 0.5 + 0.7, session.run_step_fn(step_fn))

  def test_step_fn_has_the_same_hooks_behavior_without_recovery(self):
    with ops.Graph().as_default():
      var = resource_variable_ops.ResourceVariable(0.0)

      stage_0 = state_ops.assign_add(var, 0.3)
      stage_1_0 = state_ops.assign_add(var, 0.7)
      with ops.control_dependencies([stage_1_0]):
        stage_1_1 = state_ops.assign_add(var, 0.5)
      stage_2 = state_ops.assign_add(var, 1.1)

      class Hook(session_run_hook.SessionRunHook):

        def __init__(self, testing):
          self._testing = testing

        def before_run(self, run_context):
          return session_run_hook.SessionRunArgs(fetches=stage_1_0)

        def after_run(self, run_context, run_values):
          self._testing.assertNear(0.3 + 0.5 + 0.7,
                                   run_context.session.run(var), 0.1)
          self._testing.assertNear(0.3 + 0.5 + 0.7 + 1.1,
                                   run_context.session.run(stage_2), 0.1)

      def step_fn(step_context):
        self.assertNear(0.3, step_context.session.run(stage_0), 0.1)
        return step_context.run_with_hooks(fetches=stage_1_1)

      with monitored_session.SingularMonitoredSession(
          hooks=[Hook(self)]) as session:
        self.assertEqual(0.3 + 0.5 + 0.7, session.run_step_fn(step_fn))


  def test_step_fn_exception_from_before_run(self):
    trace_the_exception = {'run_already': False, 'side_effect_counter': 0}

    with ops.Graph().as_default():
      c = array_ops.placeholder(dtypes.float32)
      v = array_ops.identity(c)
      vv = constant_op.constant(3.2)
      graph_state = variables.VariableV1(0.0)
      graph_side_effect = state_ops.assign_add(graph_state, 0.31)

      class Hook(session_run_hook.SessionRunHook):

        def __init__(self, testing):
          self._testing = testing

        def before_run(self, run_context):
          if not trace_the_exception['run_already']:
            trace_the_exception['run_already'] = True
            raise errors_impl.AbortedError(None, None, 'Abort')
          return session_run_hook.SessionRunArgs(fetches=vv)

        def after_run(self, run_context, run_values):
          self._testing.assertNear(3.2, run_values.results, 0.1)

      def step_fn(step_context):
        trace_the_exception['side_effect_counter'] += 1
        step_context.session.run(graph_side_effect)
        return step_context.run_with_hooks(fetches=v, feed_dict={c: 1.3})

      with self.cached_session() as test_session:
        with monitored_session.MonitoredSession(
            CountingSessionCreator(test_session),
            hooks=[Hook(self)]) as session:
          test_session.run(variables.global_variables_initializer())
          self.assertNear(1.3, session.run_step_fn(step_fn), 0.1)
          self.assertEqual(2, trace_the_exception['side_effect_counter'])
          self.assertNear(0.62, session.run(graph_state), 0.1)



if __name__ == '__main__':
  test.main()
