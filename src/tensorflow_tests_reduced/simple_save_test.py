import os

from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import simple_save
from tensorflow.python.saved_model import tag_constants
from ..utils.timer_wrapper import tensorflow_op_timer


class SimpleSaveTest(test.TestCase):

  def _init_and_validate_variable(self, variable_name, variable_value):
    v = variables.Variable(variable_value, name=variable_name)
    self.evaluate(variables.global_variables_initializer())
    self.assertEqual(variable_value, self.evaluate(v))
    return v

  def _check_variable_info(self, actual_variable, expected_variable):
    self.assertEqual(actual_variable.name, expected_variable.name)
    self.assertEqual(actual_variable.dtype, expected_variable.dtype)
    self.assertEqual(len(actual_variable.shape), len(expected_variable.shape))
    for i in range(len(actual_variable.shape)):
      self.assertEqual(actual_variable.shape[i], expected_variable.shape[i])

  def _check_tensor_info(self, actual_tensor_info, expected_tensor):
    self.assertEqual(actual_tensor_info.name, expected_tensor.name)
    self.assertEqual(actual_tensor_info.dtype, expected_tensor.dtype)
    self.assertEqual(
        len(actual_tensor_info.tensor_shape.dim), len(expected_tensor.shape))
    for i in range(len(actual_tensor_info.tensor_shape.dim)):
      self.assertEqual(actual_tensor_info.tensor_shape.dim[i].size,
                       expected_tensor.shape[i])

  def testSimpleSave(self):
    """Test simple_save that uses the default parameters."""
    export_dir = os.path.join(test.get_temp_dir(),
                              "test_simple_save")

    # Force the test to run in graph mode.
    # This tests a deprecated v1 API that both requires a session and uses
    # functionality that does not work with eager tensors (such as
    # build_tensor_info as called by predict_signature_def).
    with ops.Graph().as_default():
      # Initialize input and output variables and save a prediction graph using
      # the default parameters.
      with self.session(graph=ops.Graph()) as sess:
        var_x = self._init_and_validate_variable("var_x", 1)
        var_y = self._init_and_validate_variable("var_y", 2)
        inputs = {"x": var_x}
        outputs = {"y": var_y}
        timer = tensorflow_op_timer()
        with timer:
          simple_save.simple_save(sess, export_dir, inputs, outputs)
          timer.gen.send(simple_save)

      # Restore the graph with a valid tag and check the global variables and
      # signature def map.
      with self.session(graph=ops.Graph()) as sess:
        graph = loader.load(sess, [tag_constants.SERVING], export_dir)
        collection_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)

        # Check value and metadata of the saved variables.
        self.assertEqual(len(collection_vars), 2)
        self.assertEqual(1, collection_vars[0].eval())
        self.assertEqual(2, collection_vars[1].eval())
        self._check_variable_info(collection_vars[0], var_x)
        self._check_variable_info(collection_vars[1], var_y)

        # Check that the appropriate signature_def_map is created with the
        # default key and method name, and the specified inputs and outputs.
        signature_def_map = graph.signature_def
        self.assertEqual(1, len(signature_def_map))
        self.assertEqual(signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
                         list(signature_def_map.keys())[0])

        signature_def = signature_def_map[
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        self.assertEqual(signature_constants.PREDICT_METHOD_NAME,
                         signature_def.method_name)

        self.assertEqual(1, len(signature_def.inputs))
        self._check_tensor_info(signature_def.inputs["x"], var_x)
        self.assertEqual(1, len(signature_def.outputs))
        self._check_tensor_info(signature_def.outputs["y"], var_y)


if __name__ == "__main__":
  test.main()
