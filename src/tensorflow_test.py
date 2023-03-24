import pytest
from setup_tf_tpu import initialize_tpu
import sys
from tensorflow.python.platform import test
import contextlib
from tensorflow.python.eager import context
import os
import tensorflow as tf
# initialize_tpu()
# tf.config.run_functions_eagerly(True)

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

# test.TestCase._constrain_devices_and_set_default = _constrain_devices_and_set_default

retcode = pytest.main(["-x", sys.argv[1], "-s"])