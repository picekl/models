# This file is MACHINE GENERATED! Do not edit.
# Generated by: tensorflow/python/tools/api/generator/create_python_api.py script.
"""Public API for tf.debugging namespace.
"""

from __future__ import print_function as _print_function

import sys as _sys

from tensorflow.python.eager.context import get_log_device_placement
from tensorflow.python.eager.context import set_log_device_placement
from tensorflow.python.ops.check_ops import assert_equal
from tensorflow.python.ops.check_ops import assert_greater
from tensorflow.python.ops.check_ops import assert_greater_equal
from tensorflow.python.ops.check_ops import assert_integer
from tensorflow.python.ops.check_ops import assert_less
from tensorflow.python.ops.check_ops import assert_less_equal
from tensorflow.python.ops.check_ops import assert_near
from tensorflow.python.ops.check_ops import assert_negative
from tensorflow.python.ops.check_ops import assert_non_negative
from tensorflow.python.ops.check_ops import assert_non_positive
from tensorflow.python.ops.check_ops import assert_none_equal
from tensorflow.python.ops.check_ops import assert_positive
from tensorflow.python.ops.check_ops import assert_proper_iterable
from tensorflow.python.ops.check_ops import assert_rank
from tensorflow.python.ops.check_ops import assert_rank_at_least
from tensorflow.python.ops.check_ops import assert_rank_in
from tensorflow.python.ops.check_ops import assert_same_float_dtype
from tensorflow.python.ops.check_ops import assert_scalar
from tensorflow.python.ops.check_ops import assert_shapes
from tensorflow.python.ops.check_ops import assert_type
from tensorflow.python.ops.check_ops import is_non_decreasing
from tensorflow.python.ops.check_ops import is_numeric_tensor
from tensorflow.python.ops.check_ops import is_strictly_increasing
from tensorflow.python.ops.control_flow_ops import Assert
from tensorflow.python.ops.gen_array_ops import check_numerics
from tensorflow.python.ops.gen_math_ops import is_finite
from tensorflow.python.ops.gen_math_ops import is_inf
from tensorflow.python.ops.gen_math_ops import is_nan
from tensorflow.python.ops.numerics import verify_tensor_all_finite as assert_all_finite

del _print_function

from tensorflow.python.util import module_wrapper as _module_wrapper

if not isinstance(_sys.modules[__name__], _module_wrapper.TFModuleWrapper):
  _sys.modules[__name__] = _module_wrapper.TFModuleWrapper(
      _sys.modules[__name__], "debugging", public_apis=None, deprecation=True,
      has_lite=False)
