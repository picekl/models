# This file is MACHINE GENERATED! Do not edit.
# Generated by: tensorflow/python/tools/api/generator/create_python_api.py script.
"""Exception types for TensorFlow errors.
"""

from __future__ import print_function as _print_function

import sys as _sys

from tensorflow.python.framework.errors_impl import ABORTED
from tensorflow.python.framework.errors_impl import ALREADY_EXISTS
from tensorflow.python.framework.errors_impl import AbortedError
from tensorflow.python.framework.errors_impl import AlreadyExistsError
from tensorflow.python.framework.errors_impl import CANCELLED
from tensorflow.python.framework.errors_impl import CancelledError
from tensorflow.python.framework.errors_impl import DATA_LOSS
from tensorflow.python.framework.errors_impl import DEADLINE_EXCEEDED
from tensorflow.python.framework.errors_impl import DataLossError
from tensorflow.python.framework.errors_impl import DeadlineExceededError
from tensorflow.python.framework.errors_impl import FAILED_PRECONDITION
from tensorflow.python.framework.errors_impl import FailedPreconditionError
from tensorflow.python.framework.errors_impl import INTERNAL
from tensorflow.python.framework.errors_impl import INVALID_ARGUMENT
from tensorflow.python.framework.errors_impl import InternalError
from tensorflow.python.framework.errors_impl import InvalidArgumentError
from tensorflow.python.framework.errors_impl import NOT_FOUND
from tensorflow.python.framework.errors_impl import NotFoundError
from tensorflow.python.framework.errors_impl import OK
from tensorflow.python.framework.errors_impl import OUT_OF_RANGE
from tensorflow.python.framework.errors_impl import OpError
from tensorflow.python.framework.errors_impl import OutOfRangeError
from tensorflow.python.framework.errors_impl import PERMISSION_DENIED
from tensorflow.python.framework.errors_impl import PermissionDeniedError
from tensorflow.python.framework.errors_impl import RESOURCE_EXHAUSTED
from tensorflow.python.framework.errors_impl import ResourceExhaustedError
from tensorflow.python.framework.errors_impl import UNAUTHENTICATED
from tensorflow.python.framework.errors_impl import UNAVAILABLE
from tensorflow.python.framework.errors_impl import UNIMPLEMENTED
from tensorflow.python.framework.errors_impl import UNKNOWN
from tensorflow.python.framework.errors_impl import UnauthenticatedError
from tensorflow.python.framework.errors_impl import UnavailableError
from tensorflow.python.framework.errors_impl import UnimplementedError
from tensorflow.python.framework.errors_impl import UnknownError
from tensorflow.python.framework.errors_impl import error_code_from_exception_type
from tensorflow.python.framework.errors_impl import exception_type_from_error_code
from tensorflow.python.framework.errors_impl import raise_exception_on_not_ok_status

del _print_function

from tensorflow.python.util import module_wrapper as _module_wrapper

if not isinstance(_sys.modules[__name__], _module_wrapper.TFModuleWrapper):
  _sys.modules[__name__] = _module_wrapper.TFModuleWrapper(
      _sys.modules[__name__], "compat.v1.errors", public_apis=None, deprecation=False,
      has_lite=False)
