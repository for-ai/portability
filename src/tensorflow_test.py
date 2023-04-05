import pytest
import sys
from tensorflow.python.platform import test
import contextlib
from tensorflow.python.eager import context
import os
import tensorflow as tf
# initialize_tpu()


global disable_monkeypatch
disable_monkeypatch = False
def set_global_device():
    global global_device
    if os.environ['DEVICE'] == "tpu":
        global_device = "/device:TPU:0"
    elif os.environ['DEVICE'] == "gpu":
        global_device = "/device:GPU:0"
    else:
        global_device = "/device:CPU:0"

@contextlib.contextmanager
def device_context():
    with tf.device(global_device):
        yield
# Save the original as_default method
set_global_device()
_original_as_default = tf.Graph.as_default

@contextlib.contextmanager
def _constrain_devices_and_set_default(self, sess, use_gpu, force_gpu):
     """Set the session and its graph to global default and constrain devices."""
    #  print("***DEVICE CHOICE***")
     # print("*** running")
     if context.executing_eagerly():
        yield None

     else:
        with sess.graph.as_default(), sess.as_default():
            # Use the name of an actual device if one is detected, or
            # '/device:GPU:0' otherwise
            # device_name = self.gpu_device_name()
            # print("***DEVICE CHOICE")
            if os.environ['DEVICE'] == "tpu":
                device_name = "/device:TPU:0"
            elif os.environ['DEVICE'] == "gpu":
                device_name = "/device:GPU:0"
            else:
                device_name = "/device:CPU:0"
            with sess.graph.device(device_name):
                yield sess

test.TestCase._constrain_devices_and_set_default = _constrain_devices_and_set_default

@contextlib.contextmanager
def custom_as_default(self, include_device=True):
    print("***DISABLE MONKEYPATCh", disable_monkeypatch)
    # Create a context manager using the original as_default method

    # print("***CONTEXT")
    if getattr(self, '_custom_device_set', False) or not include_device or getattr(self, '_custom_device_set', False):
        self._include_device = False
        # if not include_device:
        #     self._not_include_device = False
        # print("***SKIP")
        with _original_as_default(self):
            yield
    else:
        self._custom_device_set = True

        # print("GLOBAL DEVICE", global_device)
        # Create a device context manager

        # Combine the two context managers

        with _original_as_default(self):
            with device_context():
                yield
    # class CombinedContext:
    #     def __enter__(self):
    #         original_context.__enter__()
    #         device_context().__enter__()
    #         print("HERE", global_device)

    #     def __exit__(self, exc_type, exc_val, exc_tb):
    #         device_context().__exit__(exc_type, exc_val, exc_tb)
    #         original_context.__exit__(exc_type, exc_val, exc_tb)
    #         print("THERE")

    # return CombinedContext()

# Monkey patch the Graph class
# tf.Graph.as_default = custom_as_default

if __name__ == "__main__":
    retcode = pytest.main([sys.argv[1], "-s"])
#  "-k", "test_binary_cwise_ops"