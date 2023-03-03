import tensorflow as tf
import os


def use_environment_device():
    if os.environ['DEVICE'] == "tpu":
        with tf.device('/TPU:0'):
            yield
    elif os.environ['DEVICE'] == "gpu":
        with tf.device('/GPU:0'):
            yield
    else:
        yield
