# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.GrpcServer."""

import time

import numpy as np

from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import input as input_ops
from tensorflow.python.training import queue_runner_impl
from tensorflow.python.training import server_lib


class ServerDefTest(test.TestCase):

    def testLocalServer(self):
        cluster_def = server_lib.ClusterSpec({
            "local": ["localhost:2222"]
        }).as_cluster_def()
        server_def = tensorflow_server_pb2.ServerDef(
            cluster=cluster_def, job_name="local", task_index=0, protocol="grpc")

        self.assertProtoEquals("""
    cluster {
      job { name: 'local' tasks { key: 0 value: 'localhost:2222' } }
    }
    job_name: 'local' task_index: 0 protocol: 'grpc'
    """, server_def)

        # Verifies round trip from Proto->Spec->Proto is correct.
        cluster_spec = server_lib.ClusterSpec(cluster_def)
        self.assertProtoEquals(cluster_def, cluster_spec.as_cluster_def())

    def testTwoProcesses(self):
        cluster_def = server_lib.ClusterSpec({
            "local": ["localhost:2222", "localhost:2223"]
        }).as_cluster_def()
        server_def = tensorflow_server_pb2.ServerDef(
            cluster=cluster_def, job_name="local", task_index=1, protocol="grpc")

        self.assertProtoEquals("""
    cluster {
      job { name: 'local' tasks { key: 0 value: 'localhost:2222' }
                          tasks { key: 1 value: 'localhost:2223' } }
    }
    job_name: 'local' task_index: 1 protocol: 'grpc'
    """, server_def)

        # Verifies round trip from Proto->Spec->Proto is correct.
        cluster_spec = server_lib.ClusterSpec(cluster_def)
        self.assertProtoEquals(cluster_def, cluster_spec.as_cluster_def())

    def testTwoJobs(self):
        cluster_def = server_lib.ClusterSpec({
            "ps": ["ps0:2222", "ps1:2222"],
            "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
        }).as_cluster_def()
        server_def = tensorflow_server_pb2.ServerDef(
            cluster=cluster_def, job_name="worker", task_index=2, protocol="grpc")

        self.assertProtoEquals("""
    cluster {
      job { name: 'ps' tasks { key: 0 value: 'ps0:2222' }
                       tasks { key: 1 value: 'ps1:2222' } }
      job { name: 'worker' tasks { key: 0 value: 'worker0:2222' }
                           tasks { key: 1 value: 'worker1:2222' }
                           tasks { key: 2 value: 'worker2:2222' } }
    }
    job_name: 'worker' task_index: 2 protocol: 'grpc'
    """, server_def)

        # Verifies round trip from Proto->Spec->Proto is correct.
        cluster_spec = server_lib.ClusterSpec(cluster_def)
        self.assertProtoEquals(cluster_def, cluster_spec.as_cluster_def())

    def testDenseAndSparseJobs(self):
        cluster_def = server_lib.ClusterSpec({
            "ps": ["ps0:2222", "ps1:2222"],
            "worker": {
                0: "worker0:2222",
                2: "worker2:2222"
            }
        }).as_cluster_def()
        server_def = tensorflow_server_pb2.ServerDef(
            cluster=cluster_def, job_name="worker", task_index=2, protocol="grpc")

        self.assertProtoEquals("""
    cluster {
      job { name: 'ps' tasks { key: 0 value: 'ps0:2222' }
                       tasks { key: 1 value: 'ps1:2222' } }
      job { name: 'worker' tasks { key: 0 value: 'worker0:2222' }
                           tasks { key: 2 value: 'worker2:2222' } }
    }
    job_name: 'worker' task_index: 2 protocol: 'grpc'
    """, server_def)

        # Verifies round trip from Proto->Spec->Proto is correct.
        cluster_spec = server_lib.ClusterSpec(cluster_def)
        self.assertProtoEquals(cluster_def, cluster_spec.as_cluster_def())


if __name__ == "__main__":
    test.main()
