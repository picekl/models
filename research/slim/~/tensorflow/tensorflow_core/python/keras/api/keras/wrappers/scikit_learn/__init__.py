# This file is MACHINE GENERATED! Do not edit.
# Generated by: tensorflow/python/tools/api/generator/create_python_api.py script.
"""Wrapper for using the Scikit-Learn API with Keras models.

"""

from __future__ import print_function as _print_function

import sys as _sys

from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

del _print_function

from tensorflow.python.util import module_wrapper as _module_wrapper

if not isinstance(_sys.modules[__name__], _module_wrapper.TFModuleWrapper):
  _sys.modules[__name__] = _module_wrapper.TFModuleWrapper(
      _sys.modules[__name__], "keras.wrappers.scikit_learn", public_apis=None, deprecation=True,
      has_lite=False)
