# This file is MACHINE GENERATED! Do not edit.
# Generated by: tensorflow/python/tools/api/generator/create_python_api.py script.
"""Public API for tf.config.experimental namespace.
"""

from __future__ import print_function as _print_function

import sys as _sys

from tensorflow.python.eager.context import VirtualDeviceConfiguration
from tensorflow.python.framework.config import get_device_policy
from tensorflow.python.framework.config import get_memory_growth
from tensorflow.python.framework.config import get_synchronous_execution
from tensorflow.python.framework.config import get_virtual_device_configuration
from tensorflow.python.framework.config import get_visible_devices
from tensorflow.python.framework.config import list_logical_devices
from tensorflow.python.framework.config import list_physical_devices
from tensorflow.python.framework.config import set_device_policy
from tensorflow.python.framework.config import set_memory_growth
from tensorflow.python.framework.config import set_synchronous_execution
from tensorflow.python.framework.config import set_virtual_device_configuration
from tensorflow.python.framework.config import set_visible_devices

del _print_function

from tensorflow.python.util import module_wrapper as _module_wrapper

if not isinstance(_sys.modules[__name__], _module_wrapper.TFModuleWrapper):
  _sys.modules[__name__] = _module_wrapper.TFModuleWrapper(
      _sys.modules[__name__], "config.experimental", public_apis=None, deprecation=True,
      has_lite=False)
