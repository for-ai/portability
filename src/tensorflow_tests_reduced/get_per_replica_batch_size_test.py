from absl.testing import parameterized

from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute.cluster_resolver import SimpleClusterResolver
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib
from tensorflow.python.util import nest
from ..utils.timer_wrapper import tensorflow_op_timer


class InputContextTest(test.TestCase):

    def testProperties(self):
        input_context = distribute_lib.InputContext(
            num_input_pipelines=2, input_pipeline_id=1, num_replicas_in_sync=6)
        self.assertEqual(6, input_context.num_replicas_in_sync)
        self.assertEqual(1, input_context.input_pipeline_id)
        self.assertEqual(2, input_context.num_input_pipelines)

    def testPerReplicaBatchSize(self):
        input_context = distribute_lib.InputContext(
            num_input_pipelines=2, input_pipeline_id=1, num_replicas_in_sync=6)
        timer = tensorflow_op_timer()
        with timer:
            test = input_context.get_per_replica_batch_size(12)
            timer.gen.send(test)
        self.assertEqual(2, input_context.get_per_replica_batch_size(12))
        with self.assertRaises(ValueError):
            input_context.get_per_replica_batch_size(13)

    def testStr(self):
        input_context = distribute_lib.InputContext(
            num_input_pipelines=1, input_pipeline_id=0, num_replicas_in_sync=42)
        self.assertEqual(
            "tf.distribute.InputContext(input pipeline id 0, total: 1)",
            str(input_context))
        input_context = distribute_lib.InputContext(
            num_input_pipelines=3, input_pipeline_id=1, num_replicas_in_sync=42)
        self.assertEqual(
            "tf.distribute.InputContext(input pipeline id 1, total: 3)",
            str(input_context))


if __name__ == "__main__":
    test.main()
