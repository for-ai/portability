import numpy as np

from tensorflow.lite.python import convert
from tensorflow.lite.python import op_hint
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework.graph_util_impl import _bfs_for_reachable_nodes
from tensorflow.python.framework.graph_util_impl import _extract_graph_summary
from tensorflow.python.framework.graph_util_impl import _node_name
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class ConvertTest(test_util.TensorFlowTestCase):

  def testBasic(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Try running on valid graph
    tflite_model = convert.convert_graphdef(
        sess.graph_def, input_tensors=[in_tensor], output_tensors=[out_tensor])
    self.assertTrue(tflite_model)

  def testQuantization(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      out_tensor = array_ops.fake_quant_with_min_max_args(
          in_tensor + in_tensor, min=0., max=1.)
      sess = session.Session()

    tflite_model = convert.convert_graphdef(
        sess.graph_def,
        input_tensors=[in_tensor],
        output_tensors=[out_tensor],
        inference_type=dtypes.uint8,
        quantized_input_stats=[(0., 1.)])
    self.assertTrue(tflite_model)

  def testGraphDefBasic(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32, name="input")
      _ = in_tensor + in_tensor
      sess = session.Session()

    tflite_model = convert.convert_graphdef_with_arrays(
        sess.graph_def,
        input_arrays_with_shape=[("input", [1, 16, 16, 3])],
        output_arrays=["add"],
        control_output_arrays=None,
        inference_type=dtypes.float32,
        enable_mlir_converter=False)
    self.assertTrue(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual("input", input_details[0]["name"])
    self.assertEqual(np.float32, input_details[0]["dtype"])
    self.assertTrue(([1, 16, 16, 3] == input_details[0]["shape"]).all())
    self.assertEqual((0., 0.), input_details[0]["quantization"])

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual("add", output_details[0]["name"])
    self.assertEqual(np.float32, output_details[0]["dtype"])
    self.assertTrue(([1, 16, 16, 3] == output_details[0]["shape"]).all())
    self.assertEqual((0., 0.), output_details[0]["quantization"])

  def testGraphDefQuantization(self):
    with ops.Graph().as_default():
      in_tensor_1 = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32, name="inputA")
      in_tensor_2 = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32, name="inputB")
      _ = array_ops.fake_quant_with_min_max_args(
          in_tensor_1 + in_tensor_2, min=0., max=1., name="output")
      sess = session.Session()

    tflite_model = convert.convert_graphdef_with_arrays(
        sess.graph_def,
        input_arrays_with_shape=[("inputA", [1, 16, 16, 3]),
                                 ("inputB", [1, 16, 16, 3])],
        output_arrays=["output"],
        control_output_arrays=None,
        inference_type=dtypes.uint8,
        quantized_input_stats=[(0., 1.), (0., 1.)],
        enable_mlir_converter=False,
    )
    self.assertTrue(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(2, len(input_details))
    self.assertEqual("inputA", input_details[0]["name"])
    self.assertEqual(np.uint8, input_details[0]["dtype"])
    self.assertTrue(([1, 16, 16, 3] == input_details[0]["shape"]).all())
    self.assertEqual((1., 0.),
                     input_details[0]["quantization"])  # scale, zero_point

    self.assertEqual("inputB", input_details[1]["name"])
    self.assertEqual(np.uint8, input_details[1]["dtype"])
    self.assertTrue(([1, 16, 16, 3] == input_details[1]["shape"]).all())
    self.assertEqual((1., 0.),
                     input_details[1]["quantization"])  # scale, zero_point

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual("output", output_details[0]["name"])
    self.assertEqual(np.uint8, output_details[0]["dtype"])
    self.assertTrue(([1, 16, 16, 3] == output_details[0]["shape"]).all())
    self.assertGreater(output_details[0]["quantization"][0], 0)  # scale

  def testGraphDefQuantizationInvalid(self):
    with ops.Graph().as_default():
      in_tensor_1 = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32, name="inputA")
      in_tensor_2 = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32, name="inputB")
      _ = array_ops.fake_quant_with_min_max_args(
          in_tensor_1 + in_tensor_2, min=0., max=1., name="output")
      sess = session.Session()

    with self.assertRaises(ValueError) as error:
      convert.convert_graphdef_with_arrays(
          sess.graph_def,
          input_arrays_with_shape=[("inputA", [1, 16, 16, 3]),
                                   ("inputB", [1, 16, 16, 3])],
          output_arrays=["output"],
          control_output_arrays=None,
          inference_type=dtypes.uint8,
          enable_mlir_converter=False)
    self.assertEqual(
        "The `quantized_input_stats` flag must be defined when either "
        "`inference_type` flag or `inference_input_type` flag is set to "
        "tf.int8 or tf.uint8.", str(error.exception))


if __name__ == "__main__":
      test.main()
