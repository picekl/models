# This file is MACHINE GENERATED! Do not edit.
# Generated by: tensorflow/python/tools/api/generator/create_python_api.py script.
"""Public API for tf.io.gfile namespace.
"""

from __future__ import print_function as _print_function

import sys as _sys

from tensorflow.python.lib.io.file_io import copy_v2 as copy
from tensorflow.python.lib.io.file_io import create_dir_v2 as mkdir
from tensorflow.python.lib.io.file_io import delete_file_v2 as remove
from tensorflow.python.lib.io.file_io import delete_recursively_v2 as rmtree
from tensorflow.python.lib.io.file_io import file_exists_v2 as exists
from tensorflow.python.lib.io.file_io import get_matching_files_v2 as glob
from tensorflow.python.lib.io.file_io import is_directory_v2 as isdir
from tensorflow.python.lib.io.file_io import list_directory_v2 as listdir
from tensorflow.python.lib.io.file_io import recursive_create_dir_v2 as makedirs
from tensorflow.python.lib.io.file_io import rename_v2 as rename
from tensorflow.python.lib.io.file_io import stat_v2 as stat
from tensorflow.python.lib.io.file_io import walk_v2 as walk
from tensorflow.python.platform.gfile import GFile

del _print_function

from tensorflow.python.util import module_wrapper as _module_wrapper

if not isinstance(_sys.modules[__name__], _module_wrapper.TFModuleWrapper):
  _sys.modules[__name__] = _module_wrapper.TFModuleWrapper(
      _sys.modules[__name__], "compat.v1.io.gfile", public_apis=None, deprecation=False,
      has_lite=False)
