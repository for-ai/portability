import tensorflow as tf
import os
from tensorflow.python.platform import test
import contextlib


class PortabilityTestCase(test.TestCase):
    @contextlib.contextmanager
    def _constrain_devices_and_set_default(self, sess, use_gpu, force_gpu):
        """Set the session and its graph to global default and constrain devices."""
        if context.executing_eagerly():
            yield None
        else:
            with sess.graph.as_default(), sess.as_default():
                # Use the name of an actual device if one is detected, or
                # '/device:GPU:0' otherwise
                device_name = self.gpu_device_name()
                if os.environ['DEVICE'] == "tpu":
                    device_name = "/device:TPU:0"
                elif os.environ['DEVICE'] == "tpu":
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
