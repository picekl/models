# This file is MACHINE GENERATED! Do not edit.
# Generated by: tensorflow/python/tools/api/generator/create_python_api.py script.
"""SavedModel builder.

Builds a SavedModel that can be saved to storage, is language neutral, and
enables systems to produce, consume, or transform TensorFlow Models.


"""

from __future__ import print_function as _print_function

import sys as _sys

from tensorflow.python.saved_model.builder_impl import SavedModelBuilder

del _print_function

from tensorflow.python.util import module_wrapper as _module_wrapper

if not isinstance(_sys.modules[__name__], _module_wrapper.TFModuleWrapper):
  _sys.modules[__name__] = _module_wrapper.TFModuleWrapper(
      _sys.modules[__name__], "compat.v1.saved_model.builder", public_apis=None, deprecation=False,
      has_lite=False)
