# This file is MACHINE GENERATED! Do not edit.
# Generated by: tensorflow/python/tools/api/generator/create_python_api.py script.
"""Public API for tf.config.optimizer namespace.
"""

from __future__ import print_function as _print_function

import sys as _sys

from tensorflow.python.framework.config import get_optimizer_experimental_options as get_experimental_options
from tensorflow.python.framework.config import get_optimizer_jit as get_jit
from tensorflow.python.framework.config import set_optimizer_experimental_options as set_experimental_options
from tensorflow.python.framework.config import set_optimizer_jit as set_jit

del _print_function

from tensorflow.python.util import module_wrapper as _module_wrapper

if not isinstance(_sys.modules[__name__], _module_wrapper.TFModuleWrapper):
  _sys.modules[__name__] = _module_wrapper.TFModuleWrapper(
      _sys.modules[__name__], "compat.v2.config.optimizer", public_apis=None, deprecation=False,
      has_lite=False)
