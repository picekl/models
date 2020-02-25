# This file is MACHINE GENERATED! Do not edit.
# Generated by: tensorflow/python/tools/api/generator/create_python_api.py script.
"""Ragged Tensors.

This package defines ops for manipulating ragged tensors (`tf.RaggedTensor`),
which are tensors with non-uniform shapes.  In particular, each `RaggedTensor`
has one or more *ragged dimensions*, which are dimensions whose slices may have
different lengths.  For example, the inner (column) dimension of
`rt=[[3, 1, 4, 1], [], [5, 9, 2], [6], []]` is ragged, since the column slices
(`rt[0, :]`, ..., `rt[4, :]`) have different lengths.  For a more detailed
description of ragged tensors, see the `tf.RaggedTensor` class documentation
and the [Ragged Tensor Guide](/guide/ragged_tensors).


### Additional ops that support `RaggedTensor`

Arguments that accept `RaggedTensor`s are marked in **bold**.

* `tf.batch_gather`(**params**, **indices**, name=`None`)
* `tf.bitwise.bitwise_and`(**x**, **y**, name=`None`)
* `tf.bitwise.bitwise_or`(**x**, **y**, name=`None`)
* `tf.bitwise.bitwise_xor`(**x**, **y**, name=`None`)
* `tf.bitwise.invert`(**x**, name=`None`)
* `tf.bitwise.left_shift`(**x**, **y**, name=`None`)
* `tf.bitwise.right_shift`(**x**, **y**, name=`None`)
* `tf.cast`(**x**, dtype, name=`None`)
* `tf.clip_by_value`(**t**, clip_value_min, clip_value_max, name=`None`)
* `tf.concat`(**values**, axis, name=`'concat'`)
* `tf.debugging.check_numerics`(**tensor**, message, name=`None`)
* `tf.dtypes.complex`(**real**, **imag**, name=`None`)
* `tf.dtypes.saturate_cast`(**value**, dtype, name=`None`)
* `tf.dynamic_partition`(**data**, **partitions**, num_partitions, name=`None`)
* `tf.expand_dims`(**input**, axis=`None`, name=`None`, dim=`None`)
* `tf.gather_nd`(**params**, **indices**, name=`None`, batch_dims=`0`)
* `tf.gather`(**params**, **indices**, validate_indices=`None`, name=`None`, axis=`None`, batch_dims=`0`)
* `tf.identity`(**input**, name=`None`)
* `tf.io.decode_base64`(**input**, name=`None`)
* `tf.io.decode_compressed`(**bytes**, compression_type=`''`, name=`None`)
* `tf.io.encode_base64`(**input**, pad=`False`, name=`None`)
* `tf.math.abs`(**x**, name=`None`)
* `tf.math.acos`(**x**, name=`None`)
* `tf.math.acosh`(**x**, name=`None`)
* `tf.math.add_n`(**inputs**, name=`None`)
* `tf.math.add`(**x**, **y**, name=`None`)
* `tf.math.angle`(**input**, name=`None`)
* `tf.math.asin`(**x**, name=`None`)
* `tf.math.asinh`(**x**, name=`None`)
* `tf.math.atan2`(**y**, **x**, name=`None`)
* `tf.math.atan`(**x**, name=`None`)
* `tf.math.atanh`(**x**, name=`None`)
* `tf.math.ceil`(**x**, name=`None`)
* `tf.math.conj`(**x**, name=`None`)
* `tf.math.cos`(**x**, name=`None`)
* `tf.math.cosh`(**x**, name=`None`)
* `tf.math.digamma`(**x**, name=`None`)
* `tf.math.divide_no_nan`(**x**, **y**, name=`None`)
* `tf.math.divide`(**x**, **y**, name=`None`)
* `tf.math.equal`(**x**, **y**, name=`None`)
* `tf.math.erf`(**x**, name=`None`)
* `tf.math.erfc`(**x**, name=`None`)
* `tf.math.exp`(**x**, name=`None`)
* `tf.math.expm1`(**x**, name=`None`)
* `tf.math.floor`(**x**, name=`None`)
* `tf.math.floordiv`(**x**, **y**, name=`None`)
* `tf.math.floormod`(**x**, **y**, name=`None`)
* `tf.math.greater_equal`(**x**, **y**, name=`None`)
* `tf.math.greater`(**x**, **y**, name=`None`)
* `tf.math.imag`(**input**, name=`None`)
* `tf.math.is_finite`(**x**, name=`None`)
* `tf.math.is_inf`(**x**, name=`None`)
* `tf.math.is_nan`(**x**, name=`None`)
* `tf.math.less_equal`(**x**, **y**, name=`None`)
* `tf.math.less`(**x**, **y**, name=`None`)
* `tf.math.lgamma`(**x**, name=`None`)
* `tf.math.log1p`(**x**, name=`None`)
* `tf.math.log_sigmoid`(**x**, name=`None`)
* `tf.math.log`(**x**, name=`None`)
* `tf.math.logical_and`(**x**, **y**, name=`None`)
* `tf.math.logical_not`(**x**, name=`None`)
* `tf.math.logical_or`(**x**, **y**, name=`None`)
* `tf.math.logical_xor`(**x**, **y**, name=`'LogicalXor'`)
* `tf.math.maximum`(**x**, **y**, name=`None`)
* `tf.math.minimum`(**x**, **y**, name=`None`)
* `tf.math.multiply`(**x**, **y**, name=`None`)
* `tf.math.negative`(**x**, name=`None`)
* `tf.math.not_equal`(**x**, **y**, name=`None`)
* `tf.math.pow`(**x**, **y**, name=`None`)
* `tf.math.real`(**input**, name=`None`)
* `tf.math.reciprocal`(**x**, name=`None`)
* `tf.math.reduce_any`(**input_tensor**, axis=`None`, keepdims=`False`, name=`None`)
* `tf.math.reduce_max`(**input_tensor**, axis=`None`, keepdims=`False`, name=`None`)
* `tf.math.reduce_mean`(**input_tensor**, axis=`None`, keepdims=`False`, name=`None`)
* `tf.math.reduce_min`(**input_tensor**, axis=`None`, keepdims=`False`, name=`None`)
* `tf.math.reduce_prod`(**input_tensor**, axis=`None`, keepdims=`False`, name=`None`)
* `tf.math.reduce_sum`(**input_tensor**, axis=`None`, keepdims=`False`, name=`None`)
* `tf.math.rint`(**x**, name=`None`)
* `tf.math.round`(**x**, name=`None`)
* `tf.math.rsqrt`(**x**, name=`None`)
* `tf.math.sign`(**x**, name=`None`)
* `tf.math.sin`(**x**, name=`None`)
* `tf.math.sinh`(**x**, name=`None`)
* `tf.math.sqrt`(**x**, name=`None`)
* `tf.math.square`(**x**, name=`None`)
* `tf.math.squared_difference`(**x**, **y**, name=`None`)
* `tf.math.subtract`(**x**, **y**, name=`None`)
* `tf.math.tan`(**x**, name=`None`)
* `tf.math.truediv`(**x**, **y**, name=`None`)
* `tf.math.unsorted_segment_max`(**data**, **segment_ids**, num_segments, name=`None`)
* `tf.math.unsorted_segment_mean`(**data**, **segment_ids**, num_segments, name=`None`)
* `tf.math.unsorted_segment_min`(**data**, **segment_ids**, num_segments, name=`None`)
* `tf.math.unsorted_segment_prod`(**data**, **segment_ids**, num_segments, name=`None`)
* `tf.math.unsorted_segment_sqrt_n`(**data**, **segment_ids**, num_segments, name=`None`)
* `tf.math.unsorted_segment_sum`(**data**, **segment_ids**, num_segments, name=`None`)
* `tf.one_hot`(**indices**, depth, on_value=`None`, off_value=`None`, axis=`None`, dtype=`None`, name=`None`)
* `tf.ones_like`(**tensor**, dtype=`None`, name=`None`, optimize=`True`)
* `tf.rank`(**input**, name=`None`)
* `tf.realdiv`(**x**, **y**, name=`None`)
* `tf.reduce_all`(**input_tensor**, axis=`None`, keepdims=`False`, name=`None`)
* `tf.size`(**input**, name=`None`, out_type=`tf.int32`)
* `tf.squeeze`(**input**, axis=`None`, name=`None`, squeeze_dims=`None`)
* `tf.stack`(**values**, axis=`0`, name=`'stack'`)
* `tf.strings.as_string`(**input**, precision=`-1`, scientific=`False`, shortest=`False`, width=`-1`, fill=`''`, name=`None`)
* `tf.strings.join`(**inputs**, separator=`''`, name=`None`)
* `tf.strings.length`(**input**, name=`None`, unit=`'BYTE'`)
* `tf.strings.reduce_join`(**inputs**, axis=`None`, keepdims=`False`, separator=`''`, name=`None`)
* `tf.strings.regex_full_match`(**input**, pattern, name=`None`)
* `tf.strings.regex_replace`(**input**, pattern, rewrite, replace_global=`True`, name=`None`)
* `tf.strings.strip`(**input**, name=`None`)
* `tf.strings.substr`(**input**, pos, len, name=`None`, unit=`'BYTE'`)
* `tf.strings.to_hash_bucket_fast`(**input**, num_buckets, name=`None`)
* `tf.strings.to_hash_bucket_strong`(**input**, num_buckets, key, name=`None`)
* `tf.strings.to_hash_bucket`(**input**, num_buckets, name=`None`)
* `tf.strings.to_hash_bucket`(**input**, num_buckets, name=`None`)
* `tf.strings.to_number`(**input**, out_type=`tf.float32`, name=`None`)
* `tf.strings.unicode_script`(**input**, name=`None`)
* `tf.tile`(**input**, multiples, name=`None`)
* `tf.truncatediv`(**x**, **y**, name=`None`)
* `tf.truncatemod`(**x**, **y**, name=`None`)
* `tf.where`(**condition**, **x**=`None`, **y**=`None`, name=`None`)
* `tf.zeros_like`(**tensor**, dtype=`None`, name=`None`, optimize=`True`)n
"""

from __future__ import print_function as _print_function

import sys as _sys

from tensorflow.python.ops.ragged.ragged_array_ops import boolean_mask
from tensorflow.python.ops.ragged.ragged_array_ops import stack_dynamic_partitions
from tensorflow.python.ops.ragged.ragged_concat_ops import stack
from tensorflow.python.ops.ragged.ragged_factory_ops import constant
from tensorflow.python.ops.ragged.ragged_factory_ops import constant_value
from tensorflow.python.ops.ragged.ragged_factory_ops import placeholder
from tensorflow.python.ops.ragged.ragged_functional_ops import map_flat_values
from tensorflow.python.ops.ragged.ragged_math_ops import range
from tensorflow.python.ops.ragged.ragged_tensor_value import RaggedTensorValue
from tensorflow.python.ops.ragged.segment_id_ops import row_splits_to_segment_ids
from tensorflow.python.ops.ragged.segment_id_ops import segment_ids_to_row_splits

del _print_function

from tensorflow.python.util import module_wrapper as _module_wrapper

if not isinstance(_sys.modules[__name__], _module_wrapper.TFModuleWrapper):
  _sys.modules[__name__] = _module_wrapper.TFModuleWrapper(
      _sys.modules[__name__], "compat.v1.ragged", public_apis=None, deprecation=False,
      has_lite=False)
