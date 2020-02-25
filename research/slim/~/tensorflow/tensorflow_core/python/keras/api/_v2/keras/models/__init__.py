# This file is MACHINE GENERATED! Do not edit.
# Generated by: tensorflow/python/tools/api/generator/create_python_api.py script.
"""Code for model cloning, plus model-related API entries.

"""

from __future__ import print_function as _print_function

import sys as _sys

from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.keras.models import clone_model
from tensorflow.python.keras.saving.model_config import model_from_config
from tensorflow.python.keras.saving.model_config import model_from_json
from tensorflow.python.keras.saving.model_config import model_from_yaml
from tensorflow.python.keras.saving.save import load_model
from tensorflow.python.keras.saving.save import save_model

del _print_function

from tensorflow.python.util import module_wrapper as _module_wrapper

if not isinstance(_sys.modules[__name__], _module_wrapper.TFModuleWrapper):
  _sys.modules[__name__] = _module_wrapper.TFModuleWrapper(
      _sys.modules[__name__], "keras.models", public_apis=None, deprecation=False,
      has_lite=False)
