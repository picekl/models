# This file is MACHINE GENERATED! Do not edit.
# Generated by: tensorflow/python/tools/api/generator/create_python_api.py script.
"""Loss operations for use in neural networks.

Note: All the losses are added to the `GraphKeys.LOSSES` collection by default.

"""

from __future__ import print_function as _print_function

import sys as _sys

from tensorflow.python.ops.losses.losses_impl import Reduction
from tensorflow.python.ops.losses.losses_impl import absolute_difference
from tensorflow.python.ops.losses.losses_impl import compute_weighted_loss
from tensorflow.python.ops.losses.losses_impl import cosine_distance
from tensorflow.python.ops.losses.losses_impl import hinge_loss
from tensorflow.python.ops.losses.losses_impl import huber_loss
from tensorflow.python.ops.losses.losses_impl import log_loss
from tensorflow.python.ops.losses.losses_impl import mean_pairwise_squared_error
from tensorflow.python.ops.losses.losses_impl import mean_squared_error
from tensorflow.python.ops.losses.losses_impl import sigmoid_cross_entropy
from tensorflow.python.ops.losses.losses_impl import softmax_cross_entropy
from tensorflow.python.ops.losses.losses_impl import sparse_softmax_cross_entropy
from tensorflow.python.ops.losses.util import add_loss
from tensorflow.python.ops.losses.util import get_losses
from tensorflow.python.ops.losses.util import get_regularization_loss
from tensorflow.python.ops.losses.util import get_regularization_losses
from tensorflow.python.ops.losses.util import get_total_loss

del _print_function

from tensorflow.python.util import module_wrapper as _module_wrapper

if not isinstance(_sys.modules[__name__], _module_wrapper.TFModuleWrapper):
  _sys.modules[__name__] = _module_wrapper.TFModuleWrapper(
      _sys.modules[__name__], "compat.v1.losses", public_apis=None, deprecation=False,
      has_lite=False)
