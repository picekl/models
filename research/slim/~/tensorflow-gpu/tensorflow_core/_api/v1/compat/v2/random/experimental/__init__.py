# This file is MACHINE GENERATED! Do not edit.
# Generated by: tensorflow/python/tools/api/generator/create_python_api.py script.
"""Public API for tf.random.experimental namespace.
"""

from __future__ import print_function as _print_function

import sys as _sys

from tensorflow.python.ops.stateful_random_ops import Generator
from tensorflow.python.ops.stateful_random_ops import create_rng_state
from tensorflow.python.ops.stateful_random_ops import get_global_generator
from tensorflow.python.ops.stateful_random_ops import set_global_generator

del _print_function

from tensorflow.python.util import module_wrapper as _module_wrapper

if not isinstance(_sys.modules[__name__], _module_wrapper.TFModuleWrapper):
  _sys.modules[__name__] = _module_wrapper.TFModuleWrapper(
      _sys.modules[__name__], "compat.v2.random.experimental", public_apis=None, deprecation=False,
      has_lite=False)
