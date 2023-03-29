# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for export."""

from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import metrics as metrics_module
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model.model_utils import export_output as export_output_lib


class ExportOutputTest(test.TestCase):

  def test_regress_value_must_be_float(self):
    with context.graph_mode():
      value = array_ops.placeholder(dtypes.string, 1, name='output-tensor-1')
      with self.assertRaisesRegex(
          ValueError, 'Regression output value must be a float32 Tensor'):
        export_output_lib.RegressionOutput(value)

  def test_build_standardized_signature_def_regression(self):
    with context.graph_mode():
      input_tensors = {
          'input-1':
              array_ops.placeholder(
                  dtypes.string, 1, name='input-tensor-1')
      }
      value = array_ops.placeholder(dtypes.float32, 1, name='output-tensor-1')

      export_output = export_output_lib.RegressionOutput(value)
      actual_signature_def = export_output.as_signature_def(input_tensors)

      expected_signature_def = meta_graph_pb2.SignatureDef()
      shape = tensor_shape_pb2.TensorShapeProto(
          dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=1)])
      dtype_float = types_pb2.DataType.Value('DT_FLOAT')
      dtype_string = types_pb2.DataType.Value('DT_STRING')
      expected_signature_def.inputs[
          signature_constants.REGRESS_INPUTS].CopyFrom(
              meta_graph_pb2.TensorInfo(name='input-tensor-1:0',
                                        dtype=dtype_string,
                                        tensor_shape=shape))
      expected_signature_def.outputs[
          signature_constants.REGRESS_OUTPUTS].CopyFrom(
              meta_graph_pb2.TensorInfo(name='output-tensor-1:0',
                                        dtype=dtype_float,
                                        tensor_shape=shape))

      expected_signature_def.method_name = (
          signature_constants.REGRESS_METHOD_NAME)
      self.assertEqual(actual_signature_def, expected_signature_def)


class MockSupervisedOutput(export_output_lib._SupervisedOutput):
  """So that we can test the abstract class methods directly."""

  def _get_signature_def_fn(self):
    pass


if __name__ == '__main__':
  test.main()
