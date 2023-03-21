import tensorflow as tf
import os
from tensorflow.python.platform import test
import contextlib
from tensorflow.python.eager import context
from tensorflow.python.framework import test_util

class PortabilityTestCase(test.TestCase):
  def test_session(self,
                   graph=None,
                   config=None,
                   use_gpu=True,
                   force_gpu=False):
        """Use cached_session instead."""
        print("***TEST SESSION***")
        if self.id().endswith(".test_session"):
            self.skipTest(
                "Tests that have the name \"test_session\" are automatically skipped "
                "by TensorFlow test fixture, as the name is reserved for creating "
                "sessions within tests. Please rename your test if you have a test "
                "with this name.")
        if context.executing_eagerly():
            yield None
        else:
            if graph is None:
                sess = self._get_cached_session(
                    graph, config, force_gpu, crash_if_inconsistent_args=False)
                with self._constrain_devices_and_set_default(sess, use_gpu,
                                                            force_gpu) as cached:
                    yield cached
            else:
                with self.session(graph, config, use_gpu, force_gpu) as sess:
                    yield sess

  @contextlib.contextmanager
  def session(self, graph=None, config=None, use_gpu=True, force_gpu=False):
    """A context manager for a TensorFlow Session for use in executing tests.
    Note that this will set this session and the graph as global defaults.
    Use the `use_gpu` and `force_gpu` options to control where ops are run. If
    `force_gpu` is True, all ops are pinned to `/device:GPU:0`. Otherwise, if
    `use_gpu` is True, TensorFlow tries to run as many ops on the GPU as
    possible. If both `force_gpu and `use_gpu` are False, all ops are pinned to
    the CPU.
    Example:
    ``` python
    class MyOperatorTest(test_util.TensorFlowTestCase):
      def testMyOperator(self):
        with self.session():
          valid_input = [1.0, 2.0, 3.0, 4.0, 5.0]
          result = MyOperator(valid_input).eval()
          self.assertEqual(result, [1.0, 2.0, 3.0, 5.0, 8.0]
          invalid_input = [-1.0, 2.0, 7.0]
          with self.assertRaisesOpError("negative input not supported"):
            MyOperator(invalid_input).eval()
    ```
    Args:
      graph: Optional graph to use during the returned session.
      config: An optional config_pb2.ConfigProto to use to configure the
        session.
      use_gpu: If True, attempt to run as many ops as possible on GPU.
      force_gpu: If True, pin all ops to `/device:GPU:0`.
    Yields:
      A Session object that should be used as a context manager to surround
      the graph building and execution code in a test case.
    """
    print("***SESSION***")
    if context.executing_eagerly():
        print("***EAGER***")
        if os.environ['DEVICE'] == "tpu":
            device_name = "/device:TPU:0"
        elif os.environ['DEVICE'] == "gpu":
            device_name = "/device:GPU:0"
        else:
            device_name = "/device:CPU:0"
        with tf.device(device_name):
            yield test_util.EagerSessionWarner()
    else:
      with self._create_session(graph, config, force_gpu) as sess:
        with self._constrain_devices_and_set_default(sess, use_gpu, force_gpu):
          yield sess

    @contextlib.contextmanager
    def cached_session(self,
                        graph=None,
                        config=None,
                        use_gpu=True,
                        force_gpu=False):
        """Returns a TensorFlow Session for use in executing tests.
        This method behaves differently than self.session(): for performance reasons
        `cached_session` will by default reuse the same session within the same
        test. The session returned by this function will only be closed at the end
        of the test (in the TearDown function).
        Use the `use_gpu` and `force_gpu` options to control where ops are run. If
        `force_gpu` is True, all ops are pinned to `/device:GPU:0`. Otherwise, if
        `use_gpu` is True, TensorFlow tries to run as many ops on the GPU as
        possible. If both `force_gpu and `use_gpu` are False, all ops are pinned to
        the CPU.
        Example:
        ```python
        class MyOperatorTest(test_util.TensorFlowTestCase):
        def testMyOperator(self):
            with self.cached_session() as sess:
            valid_input = [1.0, 2.0, 3.0, 4.0, 5.0]
            result = MyOperator(valid_input).eval()
            self.assertEqual(result, [1.0, 2.0, 3.0, 5.0, 8.0]
            invalid_input = [-1.0, 2.0, 7.0]
            with self.assertRaisesOpError("negative input not supported"):
                MyOperator(invalid_input).eval()
        ```
        Args:
        graph: Optional graph to use during the returned session.
        config: An optional config_pb2.ConfigProto to use to configure the
            session.
        use_gpu: If True, attempt to run as many ops as possible on GPU.
        force_gpu: If True, pin all ops to `/device:GPU:0`.
        Yields:
        A Session object that should be used as a context manager to surround
        the graph building and execution code in a test case.
        """
        print("***CACHED SESSION***")
        if context.executing_eagerly():
            yield FakeEagerSession(self)
        else:
            sess = self._get_cached_session(
                graph, config, force_gpu, crash_if_inconsistent_args=True)
            with self._constrain_devices_and_set_default(sess, use_gpu,
                                                        force_gpu) as cached:
                yield cached

    @contextlib.contextmanager
    def _constrain_devices_and_set_default(self, sess, use_gpu, force_gpu):
        """Set the session and its graph to global default and constrain devices."""
        print("***DEVICE CHOICE***")
        # print("*** running")
        if context.executing_eagerly():
            yield None
        else:

            with sess.graph.as_default(), sess.as_default():
                # Use the name of an actual device if one is detected, or
                # '/device:GPU:0' otherwise
                # device_name = self.gpu_device_name()
                print("***DEVICE CHOICE")
                if os.environ['DEVICE'] == "tpu":
                    device_name = "/device:TPU:0"
                elif os.environ['DEVICE'] == "gpu":
                    device_name = "/device:GPU:0"
                else:
                    device_name = "/device:CPU:0"
                with sess.graph.device(device_name):
                    yield sess


def use_environment_device():
    if os.environ['DEVICE'] == "tpu":
        with tf.device('/TPU:0'):
            yield
    elif os.environ['DEVICE'] == "gpu":
        with tf.device('/GPU:0'):
            yield
    else:
        yield
