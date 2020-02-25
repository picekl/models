# This file is MACHINE GENERATED! Do not edit.
# Generated by: tensorflow/python/tools/api/generator/create_python_api.py script.
"""Experimental Distribution Strategy library.
"""

from __future__ import print_function as _print_function

import sys as _sys

from tensorflow.python.distribute.central_storage_strategy import CentralStorageStrategy
from tensorflow.python.distribute.collective_all_reduce_strategy import CollectiveAllReduceStrategy as MultiWorkerMirroredStrategy
from tensorflow.python.distribute.cross_device_ops import CollectiveCommunication
from tensorflow.python.distribute.parameter_server_strategy import ParameterServerStrategy
from tensorflow.python.distribute.tpu_strategy import TPUStrategy

del _print_function

from tensorflow.python.util import module_wrapper as _module_wrapper

if not isinstance(_sys.modules[__name__], _module_wrapper.TFModuleWrapper):
  _sys.modules[__name__] = _module_wrapper.TFModuleWrapper(
      _sys.modules[__name__], "compat.v2.distribute.experimental", public_apis=None, deprecation=False,
      has_lite=False)
