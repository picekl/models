# This file is MACHINE GENERATED! Do not edit.
# Generated by: tensorflow/python/tools/api/generator/create_python_api.py script.
"""Public API for tf.lite namespace.
"""

from __future__ import print_function as _print_function

import sys as _sys

from tensorflow._api.v1.compat.v2.lite import experimental
from tensorflow.lite.python.lite import Interpreter
from tensorflow.lite.python.lite import OpsSet
from tensorflow.lite.python.lite import Optimize
from tensorflow.lite.python.lite import RepresentativeDataset
from tensorflow.lite.python.lite import TFLiteConverterV2 as TFLiteConverter
from tensorflow.lite.python.lite import TargetSpec

del _print_function

from tensorflow.python.util import module_wrapper as _module_wrapper

if not isinstance(_sys.modules[__name__], _module_wrapper.TFModuleWrapper):
  _sys.modules[__name__] = _module_wrapper.TFModuleWrapper(
      _sys.modules[__name__], "compat.v2.lite", public_apis=None, deprecation=False,
      has_lite=False)
