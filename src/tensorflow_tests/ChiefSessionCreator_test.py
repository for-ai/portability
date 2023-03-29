import glob
import os
import threading

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import debug_pb2
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util


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


def _test_dir(temp_dir, test_name):
    """Create an empty dir to use for tests.

    Args:
      temp_dir: Tmp directory path.
      test_name: Name of the test.

    Returns:
      Absolute path to the test directory.
    """
    test_dir = os.path.join(temp_dir, test_name)
    if os.path.isdir(test_dir):
        for f in glob.glob('%s/*' % test_dir):
            os.remove(f)
    else:
        os.makedirs(test_dir)
    return test_dir


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


class AbortAtNSession:
    """A mock session that aborts at the N-th run call."""

    def __init__(self, sess, n):
        self._sess = sess
        self._count = n

    def close(self):
        pass

    def run(self, *args, **kwargs):
        if self._count == 0:
            raise errors_impl.AbortedError('Aborted at N', None, None)
        self._count -= 1
        return self._sess.run(*args, **kwargs)


class StopCoordinatorWithException(session_run_hook.SessionRunHook):
    """With this hook Coordinator throws an exception after N-runs."""

    def __init__(self, calls_before_stopping, exception_to_raise=None):
        self._started_the_side_thread_already = False
        self._lock = threading.Lock()
        self._stored_exception_event = threading.Event()
        self._calls_before_stopping = calls_before_stopping
        self._exception_to_raise = (exception_to_raise or errors_impl.AbortedError(
            None, None, 'Aborted at N'))

    def _maybe_stop_with_exception(self, coord):
        while True:
            with self._lock:
                if self._calls_before_stopping == 0:
                    try:
                        raise self._exception_to_raise
                    except Exception as e:  # pylint: disable=broad-except
                        coord.request_stop(e)
                        self._stored_exception_event.set()
                        break

    def after_create_session(self, session, coord):
        if self._started_the_side_thread_already:
            return

        separate_thread = threading.Thread(
            target=self._maybe_stop_with_exception, args=(coord,))

        coord.register_thread(separate_thread)
        separate_thread.start()
        self._started_the_side_thread_already = True
        # Coordinator will take care of joining `separate_thread`.

    def after_run(self, run_context, run_values):
        stopping_now = False
        with self._lock:
            self._calls_before_stopping -= 1
            if self._calls_before_stopping == 0:
                stopping_now = True

        if stopping_now:
            self._stored_exception_event.wait()


class FailTrainingAfterCoordinatorStopped(StopCoordinatorWithException):
    """With this hook training encounters an exception after N-runs."""

    def __init__(self, calls_before_stopping):
        StopCoordinatorWithException.__init__(self, calls_before_stopping)
        self._coord = None

    def after_create_session(self, session, coord):
        self._coord = coord
        return StopCoordinatorWithException.after_create_session(
            self, session, coord)

    def after_run(self, run_context, run_values):
        StopCoordinatorWithException.after_run(self, run_context, run_values)
        try:
            # After a `run`, an exception could have been stored inside the
            # coordinator.
            self._coord.raise_requested_exception()
        except errors_impl.AbortedError:
            # In real world, the main thread may or may not know about the exception
            # that stopped the coordinator. Because the coordinator has stopped, the
            # main thread could have gotten stuck as well (for example, the
            # coordinator was supposed to execute `FIFOQueue.enqueue` while the main
            # thread is executing a blocking `FIFOQueue.dequeue`). After it got stuck,
            # the session is going to get garbage collected after some time with:
            raise errors_impl.CancelledError(None, None,
                                             'Session got garbage-collected.')


class StopCoordinatorWithException(session_run_hook.SessionRunHook):
    """With this hook Coordinator throws an exception after N-runs."""

    def __init__(self, calls_before_stopping, exception_to_raise=None):
        self._started_the_side_thread_already = False
        self._lock = threading.Lock()
        self._stored_exception_event = threading.Event()
        self._calls_before_stopping = calls_before_stopping
        self._exception_to_raise = (exception_to_raise or errors_impl.AbortedError(
            None, None, 'Aborted at N'))

    def _maybe_stop_with_exception(self, coord):
        while True:
            with self._lock:
                if self._calls_before_stopping == 0:
                    try:
                        raise self._exception_to_raise
                    except Exception as e:  # pylint: disable=broad-except
                        coord.request_stop(e)
                        self._stored_exception_event.set()
                        break

    def after_create_session(self, session, coord):
        if self._started_the_side_thread_already:
            return

        separate_thread = threading.Thread(
            target=self._maybe_stop_with_exception, args=(coord,))

        coord.register_thread(separate_thread)
        separate_thread.start()
        self._started_the_side_thread_already = True
        # Coordinator will take care of joining `separate_thread`.

    def after_run(self, run_context, run_values):
        stopping_now = False
        with self._lock:
            self._calls_before_stopping -= 1
            if self._calls_before_stopping == 0:
                stopping_now = True

        if stopping_now:
            self._stored_exception_event.wait()


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


class RecoverableSessionTest(test.TestCase):
    """_RecoverableSession tests."""

    class _SessionReturner:

        def __init__(self, sess):
            self._sess = sess

        def create_session(self):
            return self._sess

    @test_util.run_deprecated_v1
    def test_properties(self):
        with self.cached_session() as sess:
            constant_op.constant(0.0)
            recoverable_sess = monitored_session._RecoverableSession(
                self._SessionReturner(sess))
            self.assertEqual(sess.graph, recoverable_sess.graph)
            self.assertEqual(sess.sess_str, recoverable_sess.sess_str)

    @test_util.run_deprecated_v1
    def test_run(self):
        with self.cached_session() as sess:
            c = constant_op.constant(0)
            v = array_ops.identity(c)
            recoverable_sess = monitored_session._RecoverableSession(
                self._SessionReturner(sess))
            self.assertEqual(51, recoverable_sess.run(v, feed_dict={c: 51}))

    @test_util.run_deprecated_v1
    def test_recovery(self):
        with self.cached_session() as sess:

            class StackSessionCreator:

                def __init__(self, sess):
                    self.sessions_to_use = [
                        AbortAtNSession(sess, x + 1) for x in range(3)
                    ]

                def create_session(self):
                    return self.sessions_to_use.pop(0)

            c = constant_op.constant(0)
            v = array_ops.identity(c)
            session_creator = StackSessionCreator(sess)
            # List of 3 sessions to use for recovery.  The first one aborts
            # after 1 run() call, the second after 2 run calls, the third
            # after 3 run calls.
            self.assertEqual(3, len(session_creator.sessions_to_use))
            # Make the recoverable session uses these 3 sessions in sequence by
            # passing a factory that pops from the session_to_use list.
            recoverable_sess = monitored_session._RecoverableSession(
                session_creator)
            self.assertEqual(
                2, len(session_creator.sessions_to_use))  # One session popped.
            # Using first session.
            self.assertEqual(51, recoverable_sess.run(v, feed_dict={c: 51}))
            self.assertEqual(
                2, len(session_creator.sessions_to_use))  # Still 2 sessions available
            # This will fail and recover by picking up the second session.
            self.assertEqual(42, recoverable_sess.run(v, feed_dict={c: 42}))
            self.assertEqual(
                1, len(session_creator.sessions_to_use))  # Still 1 session available
            self.assertEqual(33, recoverable_sess.run(v, feed_dict={c: 33}))
            self.assertEqual(
                1, len(session_creator.sessions_to_use))  # Still 1 session available
            # This will fail and recover by picking up the last session.
            self.assertEqual(24, recoverable_sess.run(v, feed_dict={c: 24}))
            self.assertEqual(
                0, len(session_creator.sessions_to_use))  # All sessions used.
            self.assertEqual(11, recoverable_sess.run(v, feed_dict={c: 11}))
            self.assertEqual(0, recoverable_sess.run(v, feed_dict={c: 0}))
            # This will fail and throw a real error as the pop() will fail.
            with self.assertRaisesRegex(IndexError, 'pop from empty list'):
                recoverable_sess.run(v, feed_dict={c: -12})

    @test_util.run_deprecated_v1
    def test_recovery_from_coordinator_exception(self):
        with self.cached_session() as test_session:
            session_creator = CountingSessionCreator(test_session)
            session = monitored_session.MonitoredSession(
                session_creator,
                [StopCoordinatorWithException(calls_before_stopping=2)])

            self.assertEqual(1, session_creator.number_of_sessions_created)
            self.assertFalse(session.should_stop())

            c = constant_op.constant(0)
            v = array_ops.identity(c)

            # The coordinator will not abort during this call, since it's the call
            # number 0.
            self.assertEqual(51, session.run(v, feed_dict={c: 51}))
            self.assertFalse(session.should_stop())
            # The coordinator will abort during the next call, since it's the call
            # number 1.
            self.assertEqual(42, session.run(v, feed_dict={c: 42}))
            # Even though the coordinator was asked to stop, the underlying session is
            # recreated and is to be continued.
            self.assertFalse(session.should_stop())
            self.assertEqual(2, session_creator.number_of_sessions_created)

    @test_util.run_deprecated_v1
    def test_recovery_from_non_preemption_in_coordinator(self):
        with self.cached_session() as test_session:
            session_creator = CountingSessionCreator(test_session)
            hook = StopCoordinatorWithException(
                calls_before_stopping=2,
                exception_to_raise=errors_impl.UnknownError(
                    None, None, 'Some fatal exception inside the coordinator.'))
            session = monitored_session.MonitoredSession(
                session_creator, [hook])

            self.assertEqual(1, session_creator.number_of_sessions_created)
            self.assertFalse(session.should_stop())

            c = constant_op.constant(0)
            v = array_ops.identity(c)

            # The coordinator will not abort during this call, since it's the call
            # number 0.
            self.assertEqual(51, session.run(v, feed_dict={c: 51}))
            self.assertFalse(session.should_stop())
            # The coordinator will abort during the next call, since it's the call
            # number 1.
            self.assertEqual(42, session.run(v, feed_dict={c: 42}))
            # The coordinator was asked to stop due to non-redeemable error. Training
            # should stop and the session should not be recreated.
            self.assertTrue(session.should_stop())
            self.assertEqual(1, session_creator.number_of_sessions_created)
            with self.assertRaises(errors_impl.UnknownError):
                session.close()

    @test_util.run_deprecated_v1
    def test_recovery_from_session_getting_stuck(self):
        with self.cached_session() as test_session:
            session_creator = CountingSessionCreator(test_session)
            session = monitored_session.MonitoredSession(
                session_creator,
                [FailTrainingAfterCoordinatorStopped(calls_before_stopping=2)])

            self.assertEqual(1, session_creator.number_of_sessions_created)
            self.assertFalse(session.should_stop())

            c = constant_op.constant(0)
            v = array_ops.identity(c)

            # Training will not fail, since it's the call number 0.
            self.assertEqual(51, session.run(v, feed_dict={c: 51}))
            self.assertFalse(session.should_stop())
            # Training will fail during the next call, since it's the call
            # number 1.
            self.assertEqual(42, session.run(v, feed_dict={c: 42}))
            # Even though the coordinator stopped which and training failed, the
            # underlying session is recreated and training is to be continued.
            self.assertFalse(session.should_stop())
            self.assertEqual(2, session_creator.number_of_sessions_created)

    @test_util.run_deprecated_v1
    def test_step_fn_recovery_from_coordinator_exception_when_run_hooks(self):
        with self.cached_session() as test_session:
            session_creator = CountingSessionCreator(test_session)
            session = monitored_session.MonitoredSession(
                session_creator,
                [StopCoordinatorWithException(calls_before_stopping=2)])

            self.assertEqual(1, session_creator.number_of_sessions_created)
            self.assertFalse(session.should_stop())

            c = constant_op.constant(0)
            v = array_ops.identity(c)

            def feed_step_fn(value):
                def step_fn(step_context):
                    return step_context.run_with_hooks(fetches=v, feed_dict={c: value})
                return step_fn

            # The coordinator will not abort during this call, since it's the call
            # number 0.
            self.assertEqual(51, session.run_step_fn(feed_step_fn(51)))
            self.assertFalse(session.should_stop())
            # The coordinator will abort during the next call, since it's the call
            # number 1.
            self.assertEqual(42, session.run_step_fn(feed_step_fn(42)))
            # Even though the coordinator was asked to stop, the underlying session is
            # recreated and is to be continued.
            self.assertFalse(session.should_stop())
            self.assertEqual(2, session_creator.number_of_sessions_created)

    @test_util.run_deprecated_v1
    def test_recovery_from_non_preemption_in_coordinator_when_run_hooks(self):
        with self.cached_session() as test_session:
            session_creator = CountingSessionCreator(test_session)
            hook = StopCoordinatorWithException(
                calls_before_stopping=2,
                exception_to_raise=errors_impl.UnknownError(
                    None, None, 'Some fatal exception inside the coordinator.'))
            session = monitored_session.MonitoredSession(
                session_creator, [hook])

            self.assertEqual(1, session_creator.number_of_sessions_created)
            self.assertFalse(session.should_stop())

            c = constant_op.constant(0)
            v = array_ops.identity(c)

            def feed_step_fn(value):
                def step_fn(step_context):
                    return step_context.run_with_hooks(fetches=v, feed_dict={c: value})
                return step_fn

            # The coordinator will not abort during this call, since it's the call
            # number 0.
            self.assertEqual(51, session.run_step_fn(feed_step_fn(51)))
            self.assertFalse(session.should_stop())
            # The coordinator will abort during the next call, since it's the call
            # number 1.
            self.assertEqual(42, session.run_step_fn(feed_step_fn(42)))
            # The coordinator was asked to stop due to non-redeemable error. Training
            # should stop and the session should not be recreated.
            self.assertTrue(session.should_stop())
            self.assertEqual(1, session_creator.number_of_sessions_created)
            with self.assertRaises(errors_impl.UnknownError):
                session.close()

    @test_util.run_deprecated_v1
    def test_recovery_from_session_getting_stuck_when_run_hooks(self):
        with self.cached_session() as test_session:
            session_creator = CountingSessionCreator(test_session)
            session = monitored_session.MonitoredSession(
                session_creator,
                [FailTrainingAfterCoordinatorStopped(calls_before_stopping=2)])

            self.assertEqual(1, session_creator.number_of_sessions_created)
            self.assertFalse(session.should_stop())

            c = constant_op.constant(0)
            v = array_ops.identity(c)

            def feed_step_fn(value):
                def step_fn(step_context):
                    return step_context.run_with_hooks(fetches=v, feed_dict={c: value})
                return step_fn

            # Training will not fail, since it's the call number 0.
            self.assertEqual(51, session.run_step_fn(feed_step_fn(51)))
            self.assertFalse(session.should_stop())
            # Training will fail during the next call, since it's the call
            # number 1.
            self.assertEqual(42, session.run_step_fn(feed_step_fn(42)))
            # Even though the coordinator stopped which and training failed, the
            # underlying session is recreated and training is to be continued.
            self.assertFalse(session.should_stop())
            self.assertEqual(2, session_creator.number_of_sessions_created)

    def create_raw_session_with_failing_coordinator(self, session_creator, hook):
        """Return MonitoredSession that triggers coordinator failures."""
        session = monitored_session.MonitoredSession(session_creator, [hook])
        # We would like to test a situation where during fetches through the
        # raw session, the coordinator fails with an exception.  To do that, we
        # are going to use (raw_session + StopCoordinatorWithException) hook
        # combination that is stored in
        # `MonitoredSession._RecoverableSession._CoordinatedSession._sess`
        # at this point:
        session._tf_sess = lambda: session._sess._sess._sess
        # `run()` on such a session is equivalent to `run()` on the raw session
        # with separate coordinator threads independently stopping with an
        # exception.
        return session

    @test_util.run_deprecated_v1
    def test_step_fn_recovery_from_coordinator_exception_with_raw_session(self):
        with self.cached_session() as test_session:
            session_creator = CountingSessionCreator(test_session)
            session = self.create_raw_session_with_failing_coordinator(
                session_creator,
                StopCoordinatorWithException(calls_before_stopping=2))

            self.assertEqual(1, session_creator.number_of_sessions_created)
            self.assertFalse(session.should_stop())

            c = constant_op.constant(0)
            v = array_ops.identity(c)

            def feed_step_fn(value):

                def step_fn(step_context):
                    return step_context.session.run(fetches=v, feed_dict={c: value})

                return step_fn

            # The coordinator will not abort during this call, since it's the call
            # number 0.
            self.assertEqual(51, session.run_step_fn(feed_step_fn(51)))
            self.assertFalse(session.should_stop())
            # The coordinator will abort during the next call, since it's the call
            # number 1.
            self.assertEqual(42, session.run_step_fn(feed_step_fn(42)))
            # Even though the coordinator was asked to stop, the underlying session is
            # recreated and is to be continued.
            self.assertFalse(session.should_stop())
            self.assertEqual(2, session_creator.number_of_sessions_created)

    @test_util.run_deprecated_v1
    def test_recovery_from_non_preemption_in_coordinator_with_raw_session(self):
        with self.cached_session() as test_session:
            session_creator = CountingSessionCreator(test_session)
            session = self.create_raw_session_with_failing_coordinator(
                session_creator,
                StopCoordinatorWithException(
                    calls_before_stopping=2,
                    exception_to_raise=errors_impl.UnknownError(
                        None, None, 'Some fatal exception inside the coordinator.')))

            self.assertEqual(1, session_creator.number_of_sessions_created)
            self.assertFalse(session.should_stop())

            c = constant_op.constant(0)
            v = array_ops.identity(c)

            def feed_step_fn(value):

                def step_fn(step_context):
                    return step_context.run_with_hooks(fetches=v, feed_dict={c: value})

                return step_fn

            # The coordinator will not abort during this call, since it's the call
            # number 0.
            self.assertEqual(51, session.run_step_fn(feed_step_fn(51)))
            self.assertFalse(session.should_stop())
            # The coordinator will abort during the next call, since it's the call
            # number 1.
            self.assertEqual(42, session.run_step_fn(feed_step_fn(42)))
            # The coordinator was asked to stop due to non-redeemable error. Training
            # should stop and the session should not be recreated.
            self.assertTrue(session.should_stop())
            self.assertEqual(1, session_creator.number_of_sessions_created)
            with self.assertRaises(errors_impl.UnknownError):
                session.close()

    @test_util.run_deprecated_v1
    def test_recovery_from_session_getting_stuck_with_raw_session(self):
        with self.cached_session() as test_session:
            session_creator = CountingSessionCreator(test_session)
            session = self.create_raw_session_with_failing_coordinator(
                session_creator,
                FailTrainingAfterCoordinatorStopped(calls_before_stopping=2))

            self.assertEqual(1, session_creator.number_of_sessions_created)
            self.assertFalse(session.should_stop())

            c = constant_op.constant(0)
            v = array_ops.identity(c)

            def feed_step_fn(value):

                def step_fn(step_context):
                    return step_context.run_with_hooks(fetches=v, feed_dict={c: value})

                return step_fn

            # Training will not fail, since it's the call number 0.
            self.assertEqual(51, session.run_step_fn(feed_step_fn(51)))
            self.assertFalse(session.should_stop())
            # Training will fail during the next call, since it's the call
            # number 1.
            self.assertEqual(42, session.run_step_fn(feed_step_fn(42)))
            # Even though the coordinator stopped which and training failed, the
            # underlying session is recreated and training is to be continued.
            self.assertFalse(session.should_stop())
            self.assertEqual(2, session_creator.number_of_sessions_created)


class MonitoredSessionTest(test.TestCase):
    """MonitoredSession tests."""

    def test_defaults(self):
        with ops.Graph().as_default():
            a_var = variables.VariableV1(0)
            with monitored_session.MonitoredSession() as session:
                self.assertEqual(0, session.run(a_var))

    def test_last_step(self):
        logdir = _test_dir(self.get_temp_dir(), 'test_last_step')
        with ops.Graph().as_default():
            gstep = training_util.get_or_create_global_step()
            do_step = state_ops.assign_add(gstep, 1)
            # Run till step 3 and save.
            hooks = [basic_session_run_hooks.StopAtStepHook(last_step=3)]
            with monitored_session.MonitoredSession(hooks=hooks) as session:
                self.assertEqual(0, session.run(gstep))
                self.assertFalse(session.should_stop())
                self.assertEqual(1, session.run(do_step))
                self.assertFalse(session.should_stop())
                self.assertEqual(2, session.run(do_step))
                self.assertFalse(session.should_stop())
                self.assertEqual(3, session.run(do_step))
                self.assertTrue(session.should_stop())
                save_path = saver_lib._get_saver_or_default().save(
                    session._coordinated_creator.tf_sess,
                    os.path.join(logdir, 'step-3'))
            # Run till step 5 and save.

            def load_ckpt(scaffold, sess):
                scaffold.saver.restore(sess, save_path)

            session_creator = monitored_session.ChiefSessionCreator(
                monitored_session.Scaffold(init_fn=load_ckpt))
            hooks = [basic_session_run_hooks.StopAtStepHook(last_step=5)]
            with monitored_session.MonitoredSession(
                    hooks=hooks, session_creator=session_creator) as session:
                self.assertEqual(3, session.run(gstep))
                self.assertFalse(session.should_stop())
                self.assertEqual(4, session.run(do_step))
                self.assertFalse(session.should_stop())
                self.assertEqual(5, session.run(do_step))
                self.assertTrue(session.should_stop())

    def test_num_steps(self):
        logdir = _test_dir(self.get_temp_dir(), 'test_num_steps')
        with ops.Graph().as_default():
            gstep = training_util.get_or_create_global_step()
            do_step = state_ops.assign_add(gstep, 1)
            # Do 3 steps and save.
            hooks = [basic_session_run_hooks.StopAtStepHook(num_steps=3)]
            with monitored_session.MonitoredSession(hooks=hooks) as session:
                session.run(do_step)
                self.assertFalse(session.should_stop())
                session.run(do_step)
                self.assertFalse(session.should_stop())
                session.run(do_step)
                self.assertTrue(session.should_stop())
                save_path = saver_lib._get_saver_or_default().save(
                    session._coordinated_creator.tf_sess,
                    os.path.join(logdir, 'step-3'))
            # Restore and do 4 steps.

            def load_ckpt(scaffold, sess):
                scaffold.saver.restore(sess, save_path)

            session_creator = monitored_session.ChiefSessionCreator(
                scaffold=monitored_session.Scaffold(init_fn=load_ckpt))
            hooks = [basic_session_run_hooks.StopAtStepHook(num_steps=4)]
            with monitored_session.MonitoredSession(
                    hooks=hooks, session_creator=session_creator) as session:
                self.assertEqual(4, session.run(do_step))
                self.assertFalse(session.should_stop())
                session.run(do_step)
                self.assertFalse(session.should_stop())
                session.run(do_step)
                self.assertFalse(session.should_stop())
                session.run(do_step)
                self.assertTrue(session.should_stop())

    # This set of tests, verifies the supervised session behavior when exceptions
    # are raised next to the innermost session run() call.

    @test_util.run_deprecated_v1
    def test_recovery(self):
        logdir = _test_dir(self.get_temp_dir(), 'test_recovery')
        with ops.Graph().as_default():
            gstep = training_util.get_or_create_global_step()
            do_step = state_ops.assign_add(gstep, 1)
            scaffold = monitored_session.Scaffold()
            # Use a hook to save the model every 100 steps.  It also saves it at
            # the end.
            hooks = [
                basic_session_run_hooks.CheckpointSaverHook(
                    logdir, save_steps=1, scaffold=scaffold)
            ]
            with monitored_session.MonitoredSession(
                    session_creator=monitored_session.ChiefSessionCreator(
                        scaffold, checkpoint_dir=logdir),
                    hooks=hooks) as session:
                self.assertEqual(0, session.run(gstep))
                self.assertEqual(1, session.run(do_step))
                self.assertEqual(2, session.run(do_step))
            # A restart will find the checkpoint and recover automatically.
            with monitored_session.MonitoredSession(
                session_creator=monitored_session.ChiefSessionCreator(
                    scaffold, checkpoint_dir=logdir)) as session:
                self.assertEqual(2, session.run(gstep))
            # A restart will find the checkpoint and recover automatically.
            with monitored_session.MonitoredSession(
                session_creator=monitored_session.ChiefSessionCreator(
                    scaffold,
                    checkpoint_filename_with_path=checkpoint_management.
                    latest_checkpoint(logdir))) as session:
                self.assertEqual(2, session.run(gstep))

    def test_retry_initialization_on_aborted_error(self):
        # Tests that we silently retry on abort during initialization.
        with ops.Graph().as_default():
            gstep = training_util.get_or_create_global_step()
            self.init_raised_aborted_error = False

            def _init_fn(scaffold, session):
                _, _ = scaffold, session
                if not self.init_raised_aborted_error:
                    self.init_raised_aborted_error = True
                    raise errors_impl.AbortedError(None, None, 'Abort')

            with monitored_session.MonitoredSession(
                session_creator=monitored_session.ChiefSessionCreator(
                    scaffold=monitored_session.Scaffold(
                        init_fn=_init_fn))) as session:
                self.assertFalse(session.should_stop())
                self.assertEqual(0, session.run(gstep))
            self.assertTrue(self.init_raised_aborted_error)

    def _retry_test(self, ex):
        # Tests that we silently retry on error.  Note that this does not test
        # recovery as we do not use a CheckpointSaver in this test.
        with ops.Graph().as_default():
            gstep = training_util.get_or_create_global_step()
            do_step = state_ops.assign_add(gstep, 1)
            hook = RaiseOnceAtCountN(4, ex)
            with monitored_session.MonitoredSession(hooks=[hook]) as session:
                self.assertEqual(0, session.run(gstep))
                self.assertEqual(1, session.run(do_step))
                self.assertEqual(2, session.run(do_step))
                self.assertFalse(session.should_stop())
                # Here at step 3, the hook triggers and raises AbortedError.  The
                # MonitoredSession automatically retries and restart from a freshly
                # initialized session, so the step is back to 0 and running do_step
                # moves it to 1.
                self.assertEqual(1, session.run(do_step))
                self.assertFalse(session.should_stop())
                self.assertTrue(hook.raised)
                self.assertEqual(2, session.run(do_step))
                self.assertFalse(session.should_stop())

    def test_retry_on_aborted_error(self):
        self._retry_test(errors_impl.AbortedError(None, None, 'Abort'))

    def test_retry_on_unavailable_error(self):
        self._retry_test(errors_impl.UnavailableError(
            None, None, 'Unavailable'))

    def test_recover_and_retry_on_aborted_error(self):
        # Tests that we silently retry and recover on abort.  This test uses
        # a CheckpointSaver to have something to recover from.
        logdir = _test_dir(self.get_temp_dir(),
                           'test_recover_and_retry_on_aborted_error')
        with ops.Graph().as_default():
            gstep = training_util.get_or_create_global_step()
            do_step = state_ops.assign_add(gstep, 1)
            scaffold = monitored_session.Scaffold()
            abort_hook = RaiseOnceAtCountN(
                4, errors_impl.AbortedError(None, None, 'Abort'))
            # Save after each step.
            ckpt_hook = basic_session_run_hooks.CheckpointSaverHook(
                logdir, save_steps=1, scaffold=scaffold)
            hooks = [abort_hook, ckpt_hook]
            with monitored_session.MonitoredSession(
                    session_creator=monitored_session.ChiefSessionCreator(
                        scaffold, checkpoint_dir=logdir),
                    hooks=hooks) as session:
                self.assertEqual(0, session.run(gstep))
                self.assertEqual(1, session.run(do_step))
                self.assertEqual(2, session.run(do_step))
                self.assertFalse(session.should_stop())
                # Here at step 3, the hook triggers and raises AbortedError.  The
                # MonitoredSession automatically restores and retries.
                self.assertEqual(3, session.run(do_step))
                self.assertTrue(abort_hook.raised)
                self.assertFalse(session.should_stop())
                self.assertEqual(4, session.run(do_step))
                self.assertFalse(session.should_stop())

    def test_exit_cleanly_on_out_of_range_exception(self):
        # Tests that we stop cleanly when OutOfRange is raised.
        with ops.Graph().as_default():
            gstep = training_util.get_or_create_global_step()
            do_step = state_ops.assign_add(gstep, 1)
            hook = RaiseOnceAtCountN(2, errors_impl.OutOfRangeError(None, None,
                                                                    'EOI'))
            session = monitored_session.MonitoredSession(hooks=[hook])
            # session should cleanly exit from the context.
            with session:
                self.assertEqual(0, session.run(gstep))
                self.assertFalse(session.should_stop())
                # Here at step 1, the hook triggers and raises OutOfRange. The
                # session should go into should_stop() mode. It should raise the
                # exception. So next step should not be executed.
                session.run(do_step)
                self.assertTrue(False)
            self.assertTrue(session.should_stop())

    def test_exit_cleanly_on_stop_iteration_exception(self):
        # Tests that we stop cleanly when OutOfRange is raised.
        with ops.Graph().as_default():
            gstep = training_util.get_or_create_global_step()
            do_step = state_ops.assign_add(gstep, 1)
            hook = RaiseOnceAtCountN(2, StopIteration)
            session = monitored_session.MonitoredSession(hooks=[hook])
            # session should cleanly exit from the context.
            with session:
                self.assertEqual(0, session.run(gstep))
                self.assertFalse(session.should_stop())
                # Here at step 1, the hook triggers and raises StopIteration. The
                # session should go into should_stop() mode. It should raise the
                # exception. So next step should not be executed.
                session.run(do_step)
                self.assertTrue(False)
            self.assertTrue(session.should_stop())

    def test_regular_exception_pass_through_run(self):
        # Tests that regular exceptions just pass through a "with
        # MonitoredSession" block and set the session in stop mode.
        with ops.Graph().as_default():
            gstep = training_util.get_or_create_global_step()
            do_step = state_ops.assign_add(gstep, 1)
            hook = RaiseOnceAtCountN(4, RuntimeError('regular exception'))
            session = monitored_session.MonitoredSession(hooks=[hook])
            with self.assertRaisesRegex(RuntimeError, 'regular exception'):
                with session:
                    self.assertEqual(0, session.run(gstep))
                    self.assertEqual(1, session.run(do_step))
                    self.assertEqual(2, session.run(do_step))
                    self.assertFalse(session.should_stop())
                    # This triggers the hook and raises the exception
                    session.run(do_step)
                    # We should not hit this
                    self.assertFalse(True)
            self.assertTrue(hook.raised)
            self.assertTrue(session.should_stop())

    def test_regular_exception_reported_to_coord_pass_through_run(self):
        # Tests that regular exceptions reported to the coordinator from a thread
        # passes through a "run()" call within a "with MonitoredSession" block and
        # set the session in stop mode.
        with ops.Graph().as_default():
            gstep = training_util.get_or_create_global_step()
            session = monitored_session.MonitoredSession()
            run_performed_without_error = False
            with self.assertRaisesRegex(RuntimeError, 'a thread wants to stop'):
                with session:
                    self.assertEqual(0, session.run(gstep))
                    # Report an exception through the coordinator.
                    try:
                        raise RuntimeError('a thread wants to stop')
                    except RuntimeError as e:
                        session._coordinated_creator.coord.request_stop(e)
                    # Call run() which should perform normally.
                    self.assertEqual(0, session.run(gstep))
                    run_performed_without_error = True
            self.assertTrue(run_performed_without_error)

    def test_regular_exception_reported_to_coord_pass_through_return(self):
        # Tests that regular exceptions reported to the coordinator from a thread
        # passes through returning from a "with MonitoredSession" block and
        # set the session in stop mode.
        with ops.Graph().as_default():
            gstep = training_util.get_or_create_global_step()
            session = monitored_session.MonitoredSession()
            with self.assertRaisesRegex(RuntimeError, 'a thread wants to stop'):
                with session:
                    self.assertEqual(0, session.run(gstep))
                    # Report an exception through the coordinator.
                    try:
                        raise RuntimeError('a thread wants to stop')
                    except RuntimeError as e:
                        session._coordinated_creator.coord.request_stop(e)
                    self.assertTrue(session.should_stop())

    # This set of tests, verifies the session behavior when exceptions are raised
    # from code inside a "with MonitoredSession:" context.

    def test_stop_cleanly_when_no_exception_in_with_body(self):
        # Tests that regular exceptions pass through
        with ops.Graph().as_default():
            gstep = training_util.get_or_create_global_step()
            do_step = state_ops.assign_add(gstep, 1)
            session = monitored_session.MonitoredSession()
            with session:
                self.assertEqual(1, session.run(do_step))
                self.assertEqual(2, session.run(do_step))
                self.assertFalse(session.should_stop())
            # Should have closed.
            self.assertTrue(session.should_stop())
            self.assertTrue(session._is_closed())

    def test_raises_regular_exceptions_in_with_body(self):
        # Tests that regular exceptions in "with body" are seen outside.
        with ops.Graph().as_default():
            gstep = training_util.get_or_create_global_step()
            do_step = state_ops.assign_add(gstep, 1)
            session = monitored_session.MonitoredSession()
            # We should see that exception.
            with self.assertRaisesRegex(RuntimeError, 'regular exception'):
                with session:
                    self.assertEqual(1, session.run(do_step))
                    self.assertEqual(2, session.run(do_step))
                    self.assertFalse(session.should_stop())
                    # Will be visible outside the "with body".
                    raise RuntimeError('regular exception')
            # Should have closed.
            self.assertTrue(session.should_stop())
            self.assertTrue(session._is_closed())

    def test_graph(self):
        with ops.Graph().as_default() as g:
            with monitored_session.MonitoredSession() as session:
                self.assertEqual(g, session.graph)

    def test_graph_finalized_during_run_unfinalized_after_exit(self):
        with ops.Graph().as_default() as g:
            a_var = variables.VariableV1(0)
            with monitored_session.MonitoredSession() as session:
                self.assertEqual(0, session.run(a_var))
                self.assertTrue(g.finalized)
            self.assertFalse(g.finalized)

    def test_keep_finalized_graph_as_finalized(self):
        with ops.Graph().as_default() as g:
            a_var = variables.VariableV1(0)
            monitored_session.Scaffold().finalize()
            with monitored_session.MonitoredSession() as session:
                self.assertEqual(0, session.run(a_var))
                self.assertTrue(g.finalized)
            self.assertTrue(g.finalized)

    def test_merge_run_options_from_hooks(self):
        """Test for rewriting RunOptions and observing RunMetadata with hooks."""

        with ops.Graph().as_default():
            my_const = constant_op.constant(42, name='my_const')
            _ = constant_op.constant(24, name='my_const_2')

            watch_a = debug_pb2.DebugTensorWatch(
                node_name='my_const',
                output_slot=0,
                debug_ops=['DebugIdentity'],
                debug_urls=[])
            hook_a = RunOptionsMetadataHook(2, 30000, False, watch_a, False)
            watch_b = debug_pb2.DebugTensorWatch(
                node_name='my_const_2',
                output_slot=0,
                debug_ops=['DebugIdentity'],
                debug_urls=[])
            hook_b = RunOptionsMetadataHook(3, 60000, True, watch_b, True)
            with monitored_session.MonitoredSession(
                    hooks=[hook_a, hook_b]) as session:
                self.assertEqual(42, session.run(my_const))

                # trace_level=3 should have overridden trace_level=2;
                # timeout_in_ms=60000 should have overridden 30000;
                # output_partition_graphs=True should have overridden False.
                # The two debug tensor watches should have been merged.
                self.assertEqual([
                    config_pb2.RunOptions(
                        trace_level=3,
                        timeout_in_ms=60000,
                        output_partition_graphs=True,
                        debug_options=debug_pb2.DebugOptions(
                            debug_tensor_watch_opts=[watch_a, watch_b]),
                        report_tensor_allocations_upon_oom=True),
                ], hook_b.run_options_list)
                self.assertEqual(1, len(hook_b.run_metadata_list))
                self.assertTrue(
                    isinstance(hook_b.run_metadata_list[0], config_pb2.RunMetadata))
                self.assertGreater(
                    len(hook_b.run_metadata_list[0].partition_graphs), 0)

    def test_merge_caller_and_hook_run_options(self):
        """Test that RunOptions from caller and hooks can be merged properly."""

        with ops.Graph().as_default():
            my_const = constant_op.constant(42, name='my_const')
            _ = constant_op.constant(24, name='my_const_2')

            hook_watch = debug_pb2.DebugTensorWatch(
                node_name='my_const_2',
                output_slot=0,
                debug_ops=['DebugIdentity'],
                debug_urls=[])
            hook = RunOptionsMetadataHook(2, 60000, False, hook_watch, False)
            with monitored_session.MonitoredSession(hooks=[hook]) as session:
                caller_watch = debug_pb2.DebugTensorWatch(
                    node_name='my_const',
                    output_slot=0,
                    debug_ops=['DebugIdentity'],
                    debug_urls=[])
                caller_options = config_pb2.RunOptions(
                    trace_level=3,
                    timeout_in_ms=30000,
                    output_partition_graphs=True,
                    report_tensor_allocations_upon_oom=True)
                caller_options.debug_options.debug_tensor_watch_opts.extend(
                    [caller_watch])
                self.assertEqual(42, session.run(
                    my_const, options=caller_options))

                # trace_level=3 from the caller should override 2 from the hook.
                # timeout_in_ms=60000 from the hook should override from the caller.
                # output_partition_graph=True from the caller should override False
                # from the hook.
                # The two debug watches from the caller and the hook should be merged,
                # in that order.
                self.assertEqual([
                    config_pb2.RunOptions(
                        trace_level=3,
                        timeout_in_ms=60000,
                        output_partition_graphs=True,
                        debug_options=debug_pb2.DebugOptions(
                            debug_tensor_watch_opts=[caller_watch, hook_watch]),
                        report_tensor_allocations_upon_oom=True),
                ], hook.run_options_list)
                self.assertEqual(1, len(hook.run_metadata_list))
                self.assertTrue(
                    isinstance(hook.run_metadata_list[0], config_pb2.RunMetadata))
                self.assertGreater(
                    len(hook.run_metadata_list[0].partition_graphs), 0)

    @test_util.run_deprecated_v1
    def test_with_statement_and_close(self):
        # Test case for https://github.com/tensorflow/tensorflow/issues/12224
        # where close() inside the with should have a better error message.
        with self.assertRaisesRegex(RuntimeError, 'Session is already closed'):
            with monitored_session.MonitoredSession() as session:
                session.close()

    def test_step_fn_example(self):
        with ops.Graph().as_default():
            c = array_ops.placeholder(dtypes.float32)
            v = array_ops.identity(c)

            def step_fn(step_context):
                value = step_context.run_with_hooks(
                    fetches=v, feed_dict={c: 3.2})
                return value

            with monitored_session.MonitoredSession() as session:
                self.assertNear(3.2, session.run_step_fn(step_fn), 0.1)

    def test_step_function_stops(self):
        with ops.Graph().as_default():

            def step_fn(step_context):
                step_context.request_stop()

            with monitored_session.MonitoredSession() as session:
                self.assertEqual(None, session.run_step_fn(step_fn))
                self.assertTrue(session.should_stop())

    def test_step_request_stop_without_a_with_block(self):
        with ops.Graph().as_default():
            was_stop_iteration_raised = False

            def step_fn(step_context):
                step_context.request_stop()

            session = monitored_session.MonitoredSession()
            try:
                self.assertEqual(None, session.run_step_fn(step_fn))
            except StopIteration:
                was_stop_iteration_raised = True

            self.assertTrue(was_stop_iteration_raised)
            self.assertFalse(session.should_stop())

    def test_step_request_stop_in_a_loop(self):
        with ops.Graph().as_default():
            def step_fn(step_context):
                step_context.request_stop()

            with monitored_session.MonitoredSession() as session:
                while not session.should_stop():
                    _ = session.run_step_fn(step_fn)
                    self.fail(
                        'An exception should be raised on the line above.')

    def test_step_request_stop_with_returning_a_type(self):
        with ops.Graph().as_default():

            def step_fn(step_context):
                del step_context
                return 'a type'

            with monitored_session.MonitoredSession() as session:
                self.assertEqual('a type', session.run_step_fn(step_fn))

    def test_step_with_extra_arguments(self):
        with ops.Graph().as_default():

            def step_fn(step_context, extra_foo):
                del step_context, extra_foo

            with monitored_session.MonitoredSession() as session:
                with self.assertRaisesRegex(
                        ValueError,
                        '`step_fn` may either have one `step_context` argument'):
                    self.assertEqual(None, session.run_step_fn(step_fn))

    def test_step_fn_belongs_to_a_class(self):
        with ops.Graph().as_default():
            c = array_ops.placeholder(dtypes.float32)
            v = array_ops.identity(c)

            class Model:

                def step_fn(self, step_context):
                    return step_context.run_with_hooks(fetches=v, feed_dict={c: 3.2})

            with monitored_session.MonitoredSession() as session:
                model = Model()
                self.assertNear(3.2, session.run_step_fn(model.step_fn), 0.1)

    def test_step_fn_belongs_to_a_class_and_has_extra_methods(self):
        with ops.Graph().as_default():

            class Model:

                def step_fn(self, step_context, extra_foo):
                    del step_context, extra_foo

            with monitored_session.MonitoredSession() as session:
                with self.assertRaisesRegex(
                        ValueError,
                        '`step_fn` may either have one `step_context` argument'):
                    model = Model()
                    self.assertEqual(None, session.run_step_fn(model.step_fn))

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

    def test_step_fn_with_hooks_and_request_stop(self):
        with ops.Graph().as_default():
            trace_the_hook = {'before_run': False, 'after_run': False}

            class Hook(session_run_hook.SessionRunHook):

                def before_run(self, run_context):
                    trace_the_hook['before_run'] = True

                def after_run(self, run_context, run_values):
                    trace_the_hook['after_run'] = True

            def step_fn(step_context):
                step_context.request_stop()

            with monitored_session.MonitoredSession(hooks=[Hook()]) as session:
                self.assertEqual(None, session.run_step_fn(step_fn))
                self.assertTrue(session.should_stop())
                # `step_context.request_stop()` in a step_fn interrupts the flow of
                # running the hooks.
                self.assertFalse(trace_the_hook['before_run'])
                self.assertFalse(trace_the_hook['after_run'])

    def test_recovers_from_an_exception_in_step_fn(self):
        trace_the_exception = {'run_already': False}

        with ops.Graph().as_default():
            c = array_ops.placeholder(dtypes.float32)
            v = array_ops.identity(c)

            def step_fn(step_context):
                if not trace_the_exception['run_already']:
                    trace_the_exception['run_already'] = True
                    raise errors_impl.AbortedError(None, None, 'Abort')

                return step_context.run_with_hooks(fetches=v, feed_dict={c: 3.2})

            with monitored_session.MonitoredSession() as session:
                self.assertNear(3.2, session.run_step_fn(step_fn), 0.1)
            self.assertTrue(trace_the_exception['run_already'])

    def test_recovers_from_an_exception_in_step_fn_after_hooks(self):
        trace_the_exception = {'run_already': False, 'side_effect_counter': 0}

        with ops.Graph().as_default():
            c = array_ops.placeholder(dtypes.float32)
            v = array_ops.identity(c)
            graph_state = variables.VariableV1(0.0)
            graph_side_effect = state_ops.assign_add(graph_state, 0.31)

            def step_fn(step_context):
                trace_the_exception['side_effect_counter'] += 1
                step_context.session.run(graph_side_effect)

                value = step_context.run_with_hooks(
                    fetches=v, feed_dict={c: 3.2})

                if not trace_the_exception['run_already']:
                    trace_the_exception['run_already'] = True
                    raise errors_impl.AbortedError(None, None, 'Abort')

                return value

            with self.cached_session() as test_session:
                with monitored_session.MonitoredSession(
                        CountingSessionCreator(test_session)) as session:
                    session.run(variables.global_variables_initializer())

                    self.assertNear(3.2, session.run_step_fn(step_fn), 0.1)
                    self.assertTrue(trace_the_exception['run_already'])
                    # Make sure the rest of the body of the step_fn is re-executed upon
                    # AbortedError:
                    self.assertEqual(
                        2, trace_the_exception['side_effect_counter'])
                    self.assertNear(0.62, session.run(graph_state), 0.1)

    def test_step_fn_doesnt_recover_when_it_wasnt_asked_to(self):
        trace_the_exception = {'run_already': False}

        with ops.Graph().as_default():
            c = array_ops.placeholder(dtypes.float32)
            v = array_ops.identity(c)

            def step_fn(step_context):
                if not trace_the_exception['run_already']:
                    trace_the_exception['run_already'] = True
                    raise errors_impl.AbortedError(None, None, 'Abort')

                value = step_context.run_with_hooks(
                    fetches=v, feed_dict={c: 3.2})
                return value

            with monitored_session.SingularMonitoredSession() as session:
                with self.assertRaisesRegex(errors_impl.AbortedError, 'Abort'):
                    self.assertNear(3.2, session.run_step_fn(step_fn), 0.1)
                    self.fail()

            self.assertTrue(trace_the_exception['run_already'])

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
                    self.assertEqual(
                        2, trace_the_exception['side_effect_counter'])
                    self.assertNear(0.62, session.run(graph_state), 0.1)


if __name__ == '__main__':
    test.main()
