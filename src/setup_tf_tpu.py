import pytest
import os
import json
import gc
import tensorflow as tf

def initialize_tpu(): 
    # if os.environ['DEVICE'] == "tpu":
    os.environ.TPU_NAME = 'local'
    print("*** TPU_NAME", os.environ.TPU_NAME)
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    tpu_devices = tf.config.list_logical_devices('TPU')
    print("test device", tpu_devices)