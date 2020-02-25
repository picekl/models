# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Common utils for all inference plugin files."""

from tensorboard._vendor.tensorflow_serving.apis import classification_pb2
from tensorboard._vendor.tensorflow_serving.apis import regression_pb2


class InvalidUserInputError(Exception):
  """An exception to throw if user input is detected to be invalid.

  Attributes:
    original_exception: The triggering `Exception` object to be wrapped, or
      a string.
  """

  def __init__(self, original_exception):
    """Inits InvalidUserInputError."""
    self.original_exception = original_exception
    Exception.__init__(self)

  @property
  def message(self):
    return 'InvalidUserInputError: ' + str(self.original_exception)


def convert_predict_response(pred, serving_bundle):
  """Converts a PredictResponse to ClassificationResponse or RegressionResponse.

  Args:
    pred: PredictResponse to convert.
    serving_bundle: A `ServingBundle` object that contains the information about
      the serving request that the response was generated by.

  Returns:
    A ClassificationResponse or RegressionResponse.
  """
  output = pred.outputs[serving_bundle.predict_output_tensor]
  raw_output = output.float_val
  if serving_bundle.model_type == 'classification':
    values = []
    for example_index in range(output.tensor_shape.dim[0].size):
      start = example_index * output.tensor_shape.dim[1].size
      values.append(raw_output[start:start + output.tensor_shape.dim[1].size])
  else:
    values = raw_output
  return convert_prediction_values(values, serving_bundle, pred.model_spec)

def convert_prediction_values(values, serving_bundle, model_spec=None):
  """Converts tensor values into ClassificationResponse or RegressionResponse.

  Args:
    values: For classification, a 2D list of numbers. The first dimension is for
      each example being predicted. The second dimension are the probabilities
      for each class ID in the prediction. For regression, a 1D list of numbers,
      with a regression score for each example being predicted.
    serving_bundle: A `ServingBundle` object that contains the information about
      the serving request that the response was generated by.
    model_spec: Optional model spec to put into the response.

  Returns:
    A ClassificationResponse or RegressionResponse.
  """
  if serving_bundle.model_type == 'classification':
    response = classification_pb2.ClassificationResponse()
    for example_index in range(len(values)):
      classification = response.result.classifications.add()
      for class_index in range(len(values[example_index])):
        class_score = classification.classes.add()
        class_score.score = values[example_index][class_index]
        class_score.label = str(class_index)
  else:
    response = regression_pb2.RegressionResponse()
    for example_index in range(len(values)):
      regression = response.result.regressions.add()
      regression.value = values[example_index]
  if model_spec:
    response.model_spec.CopyFrom(model_spec)
  return response
