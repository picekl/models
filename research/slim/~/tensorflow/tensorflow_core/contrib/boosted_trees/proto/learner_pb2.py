# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/contrib/boosted_trees/proto/learner.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from tensorflow.contrib.boosted_trees.proto import tree_config_pb2 as tensorflow_dot_contrib_dot_boosted__trees_dot_proto_dot_tree__config__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensorflow/contrib/boosted_trees/proto/learner.proto',
  package='tensorflow.boosted_trees.learner',
  syntax='proto3',
  serialized_options=_b('\370\001\001'),
  serialized_pb=_b('\n4tensorflow/contrib/boosted_trees/proto/learner.proto\x12 tensorflow.boosted_trees.learner\x1a\x38tensorflow/contrib/boosted_trees/proto/tree_config.proto\"K\n\x18TreeRegularizationConfig\x12\n\n\x02l1\x18\x01 \x01(\x02\x12\n\n\x02l2\x18\x02 \x01(\x02\x12\x17\n\x0ftree_complexity\x18\x03 \x01(\x02\"v\n\x15TreeConstraintsConfig\x12\x16\n\x0emax_tree_depth\x18\x01 \x01(\r\x12\x17\n\x0fmin_node_weight\x18\x02 \x01(\x02\x12,\n$max_number_of_unique_feature_columns\x18\x03 \x01(\x03\"\x96\x02\n\x12LearningRateConfig\x12J\n\x05\x66ixed\x18\x01 \x01(\x0b\x32\x39.tensorflow.boosted_trees.learner.LearningRateFixedConfigH\x00\x12T\n\x07\x64ropout\x18\x02 \x01(\x0b\x32\x41.tensorflow.boosted_trees.learner.LearningRateDropoutDrivenConfigH\x00\x12U\n\x0bline_search\x18\x03 \x01(\x0b\x32>.tensorflow.boosted_trees.learner.LearningRateLineSearchConfigH\x00\x42\x07\n\x05tuner\"0\n\x17LearningRateFixedConfig\x12\x15\n\rlearning_rate\x18\x01 \x01(\x02\"L\n\x1cLearningRateLineSearchConfig\x12\x19\n\x11max_learning_rate\x18\x01 \x01(\x02\x12\x11\n\tnum_steps\x18\x02 \x01(\x05\"a\n\x0f\x41veragingConfig\x12\x1e\n\x14\x61verage_last_n_trees\x18\x01 \x01(\x02H\x00\x12$\n\x1a\x61verage_last_percent_trees\x18\x02 \x01(\x02H\x00\x42\x08\n\x06\x63onfig\"~\n\x1fLearningRateDropoutDrivenConfig\x12\x1b\n\x13\x64ropout_probability\x18\x01 \x01(\x02\x12\'\n\x1fprobability_of_skipping_dropout\x18\x02 \x01(\x02\x12\x15\n\rlearning_rate\x18\x03 \x01(\x02\"\xf9\t\n\rLearnerConfig\x12\x13\n\x0bnum_classes\x18\x01 \x01(\r\x12#\n\x19\x66\x65\x61ture_fraction_per_tree\x18\x02 \x01(\x02H\x00\x12$\n\x1a\x66\x65\x61ture_fraction_per_level\x18\x03 \x01(\x02H\x00\x12R\n\x0eregularization\x18\x04 \x01(\x0b\x32:.tensorflow.boosted_trees.learner.TreeRegularizationConfig\x12L\n\x0b\x63onstraints\x18\x05 \x01(\x0b\x32\x37.tensorflow.boosted_trees.learner.TreeConstraintsConfig\x12Q\n\x0cpruning_mode\x18\x08 \x01(\x0e\x32;.tensorflow.boosted_trees.learner.LearnerConfig.PruningMode\x12Q\n\x0cgrowing_mode\x18\t \x01(\x0e\x32;.tensorflow.boosted_trees.learner.LearnerConfig.GrowingMode\x12Q\n\x13learning_rate_tuner\x18\x06 \x01(\x0b\x32\x34.tensorflow.boosted_trees.learner.LearningRateConfig\x12`\n\x14multi_class_strategy\x18\n \x01(\x0e\x32\x42.tensorflow.boosted_trees.learner.LearnerConfig.MultiClassStrategy\x12K\n\x10\x61veraging_config\x18\x0b \x01(\x0b\x32\x31.tensorflow.boosted_trees.learner.AveragingConfig\x12Z\n\x11weak_learner_type\x18\x0c \x01(\x0e\x32?.tensorflow.boosted_trees.learner.LearnerConfig.WeakLearnerType\x12K\n\x0f\x65\x61\x63h_tree_start\x18\r \x01(\x0b\x32\x32.tensorflow.boosted_trees.trees.DecisionTreeConfig\x12\"\n\x1a\x65\x61\x63h_tree_start_num_layers\x18\x0e \x01(\x05\"J\n\x0bPruningMode\x12\x1c\n\x18PRUNING_MODE_UNSPECIFIED\x10\x00\x12\r\n\tPRE_PRUNE\x10\x01\x12\x0e\n\nPOST_PRUNE\x10\x02\"O\n\x0bGrowingMode\x12\x1c\n\x18GROWING_MODE_UNSPECIFIED\x10\x00\x12\x0e\n\nWHOLE_TREE\x10\x01\x12\x12\n\x0eLAYER_BY_LAYER\x10\x02\"v\n\x12MultiClassStrategy\x12$\n MULTI_CLASS_STRATEGY_UNSPECIFIED\x10\x00\x12\x12\n\x0eTREE_PER_CLASS\x10\x01\x12\x10\n\x0c\x46ULL_HESSIAN\x10\x02\x12\x14\n\x10\x44IAGONAL_HESSIAN\x10\x03\"H\n\x0fWeakLearnerType\x12\x18\n\x14NORMAL_DECISION_TREE\x10\x00\x12\x1b\n\x17OBLIVIOUS_DECISION_TREE\x10\x01\x42\x12\n\x10\x66\x65\x61ture_fractionB\x03\xf8\x01\x01\x62\x06proto3')
  ,
  dependencies=[tensorflow_dot_contrib_dot_boosted__trees_dot_proto_dot_tree__config__pb2.DESCRIPTOR,])



_LEARNERCONFIG_PRUNINGMODE = _descriptor.EnumDescriptor(
  name='PruningMode',
  full_name='tensorflow.boosted_trees.learner.LearnerConfig.PruningMode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='PRUNING_MODE_UNSPECIFIED', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PRE_PRUNE', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='POST_PRUNE', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1886,
  serialized_end=1960,
)
_sym_db.RegisterEnumDescriptor(_LEARNERCONFIG_PRUNINGMODE)

_LEARNERCONFIG_GROWINGMODE = _descriptor.EnumDescriptor(
  name='GrowingMode',
  full_name='tensorflow.boosted_trees.learner.LearnerConfig.GrowingMode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='GROWING_MODE_UNSPECIFIED', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='WHOLE_TREE', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LAYER_BY_LAYER', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1962,
  serialized_end=2041,
)
_sym_db.RegisterEnumDescriptor(_LEARNERCONFIG_GROWINGMODE)

_LEARNERCONFIG_MULTICLASSSTRATEGY = _descriptor.EnumDescriptor(
  name='MultiClassStrategy',
  full_name='tensorflow.boosted_trees.learner.LearnerConfig.MultiClassStrategy',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='MULTI_CLASS_STRATEGY_UNSPECIFIED', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TREE_PER_CLASS', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FULL_HESSIAN', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DIAGONAL_HESSIAN', index=3, number=3,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=2043,
  serialized_end=2161,
)
_sym_db.RegisterEnumDescriptor(_LEARNERCONFIG_MULTICLASSSTRATEGY)

_LEARNERCONFIG_WEAKLEARNERTYPE = _descriptor.EnumDescriptor(
  name='WeakLearnerType',
  full_name='tensorflow.boosted_trees.learner.LearnerConfig.WeakLearnerType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='NORMAL_DECISION_TREE', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OBLIVIOUS_DECISION_TREE', index=1, number=1,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=2163,
  serialized_end=2235,
)
_sym_db.RegisterEnumDescriptor(_LEARNERCONFIG_WEAKLEARNERTYPE)


_TREEREGULARIZATIONCONFIG = _descriptor.Descriptor(
  name='TreeRegularizationConfig',
  full_name='tensorflow.boosted_trees.learner.TreeRegularizationConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='l1', full_name='tensorflow.boosted_trees.learner.TreeRegularizationConfig.l1', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='l2', full_name='tensorflow.boosted_trees.learner.TreeRegularizationConfig.l2', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tree_complexity', full_name='tensorflow.boosted_trees.learner.TreeRegularizationConfig.tree_complexity', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=148,
  serialized_end=223,
)


_TREECONSTRAINTSCONFIG = _descriptor.Descriptor(
  name='TreeConstraintsConfig',
  full_name='tensorflow.boosted_trees.learner.TreeConstraintsConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='max_tree_depth', full_name='tensorflow.boosted_trees.learner.TreeConstraintsConfig.max_tree_depth', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='min_node_weight', full_name='tensorflow.boosted_trees.learner.TreeConstraintsConfig.min_node_weight', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_number_of_unique_feature_columns', full_name='tensorflow.boosted_trees.learner.TreeConstraintsConfig.max_number_of_unique_feature_columns', index=2,
      number=3, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=225,
  serialized_end=343,
)


_LEARNINGRATECONFIG = _descriptor.Descriptor(
  name='LearningRateConfig',
  full_name='tensorflow.boosted_trees.learner.LearningRateConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='fixed', full_name='tensorflow.boosted_trees.learner.LearningRateConfig.fixed', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dropout', full_name='tensorflow.boosted_trees.learner.LearningRateConfig.dropout', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='line_search', full_name='tensorflow.boosted_trees.learner.LearningRateConfig.line_search', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='tuner', full_name='tensorflow.boosted_trees.learner.LearningRateConfig.tuner',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=346,
  serialized_end=624,
)


_LEARNINGRATEFIXEDCONFIG = _descriptor.Descriptor(
  name='LearningRateFixedConfig',
  full_name='tensorflow.boosted_trees.learner.LearningRateFixedConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='learning_rate', full_name='tensorflow.boosted_trees.learner.LearningRateFixedConfig.learning_rate', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=626,
  serialized_end=674,
)


_LEARNINGRATELINESEARCHCONFIG = _descriptor.Descriptor(
  name='LearningRateLineSearchConfig',
  full_name='tensorflow.boosted_trees.learner.LearningRateLineSearchConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='max_learning_rate', full_name='tensorflow.boosted_trees.learner.LearningRateLineSearchConfig.max_learning_rate', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_steps', full_name='tensorflow.boosted_trees.learner.LearningRateLineSearchConfig.num_steps', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=676,
  serialized_end=752,
)


_AVERAGINGCONFIG = _descriptor.Descriptor(
  name='AveragingConfig',
  full_name='tensorflow.boosted_trees.learner.AveragingConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='average_last_n_trees', full_name='tensorflow.boosted_trees.learner.AveragingConfig.average_last_n_trees', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='average_last_percent_trees', full_name='tensorflow.boosted_trees.learner.AveragingConfig.average_last_percent_trees', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='config', full_name='tensorflow.boosted_trees.learner.AveragingConfig.config',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=754,
  serialized_end=851,
)


_LEARNINGRATEDROPOUTDRIVENCONFIG = _descriptor.Descriptor(
  name='LearningRateDropoutDrivenConfig',
  full_name='tensorflow.boosted_trees.learner.LearningRateDropoutDrivenConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='dropout_probability', full_name='tensorflow.boosted_trees.learner.LearningRateDropoutDrivenConfig.dropout_probability', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='probability_of_skipping_dropout', full_name='tensorflow.boosted_trees.learner.LearningRateDropoutDrivenConfig.probability_of_skipping_dropout', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='learning_rate', full_name='tensorflow.boosted_trees.learner.LearningRateDropoutDrivenConfig.learning_rate', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=853,
  serialized_end=979,
)


_LEARNERCONFIG = _descriptor.Descriptor(
  name='LearnerConfig',
  full_name='tensorflow.boosted_trees.learner.LearnerConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_classes', full_name='tensorflow.boosted_trees.learner.LearnerConfig.num_classes', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='feature_fraction_per_tree', full_name='tensorflow.boosted_trees.learner.LearnerConfig.feature_fraction_per_tree', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='feature_fraction_per_level', full_name='tensorflow.boosted_trees.learner.LearnerConfig.feature_fraction_per_level', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='regularization', full_name='tensorflow.boosted_trees.learner.LearnerConfig.regularization', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='constraints', full_name='tensorflow.boosted_trees.learner.LearnerConfig.constraints', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pruning_mode', full_name='tensorflow.boosted_trees.learner.LearnerConfig.pruning_mode', index=5,
      number=8, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='growing_mode', full_name='tensorflow.boosted_trees.learner.LearnerConfig.growing_mode', index=6,
      number=9, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='learning_rate_tuner', full_name='tensorflow.boosted_trees.learner.LearnerConfig.learning_rate_tuner', index=7,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='multi_class_strategy', full_name='tensorflow.boosted_trees.learner.LearnerConfig.multi_class_strategy', index=8,
      number=10, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='averaging_config', full_name='tensorflow.boosted_trees.learner.LearnerConfig.averaging_config', index=9,
      number=11, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='weak_learner_type', full_name='tensorflow.boosted_trees.learner.LearnerConfig.weak_learner_type', index=10,
      number=12, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='each_tree_start', full_name='tensorflow.boosted_trees.learner.LearnerConfig.each_tree_start', index=11,
      number=13, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='each_tree_start_num_layers', full_name='tensorflow.boosted_trees.learner.LearnerConfig.each_tree_start_num_layers', index=12,
      number=14, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _LEARNERCONFIG_PRUNINGMODE,
    _LEARNERCONFIG_GROWINGMODE,
    _LEARNERCONFIG_MULTICLASSSTRATEGY,
    _LEARNERCONFIG_WEAKLEARNERTYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='feature_fraction', full_name='tensorflow.boosted_trees.learner.LearnerConfig.feature_fraction',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=982,
  serialized_end=2255,
)

_LEARNINGRATECONFIG.fields_by_name['fixed'].message_type = _LEARNINGRATEFIXEDCONFIG
_LEARNINGRATECONFIG.fields_by_name['dropout'].message_type = _LEARNINGRATEDROPOUTDRIVENCONFIG
_LEARNINGRATECONFIG.fields_by_name['line_search'].message_type = _LEARNINGRATELINESEARCHCONFIG
_LEARNINGRATECONFIG.oneofs_by_name['tuner'].fields.append(
  _LEARNINGRATECONFIG.fields_by_name['fixed'])
_LEARNINGRATECONFIG.fields_by_name['fixed'].containing_oneof = _LEARNINGRATECONFIG.oneofs_by_name['tuner']
_LEARNINGRATECONFIG.oneofs_by_name['tuner'].fields.append(
  _LEARNINGRATECONFIG.fields_by_name['dropout'])
_LEARNINGRATECONFIG.fields_by_name['dropout'].containing_oneof = _LEARNINGRATECONFIG.oneofs_by_name['tuner']
_LEARNINGRATECONFIG.oneofs_by_name['tuner'].fields.append(
  _LEARNINGRATECONFIG.fields_by_name['line_search'])
_LEARNINGRATECONFIG.fields_by_name['line_search'].containing_oneof = _LEARNINGRATECONFIG.oneofs_by_name['tuner']
_AVERAGINGCONFIG.oneofs_by_name['config'].fields.append(
  _AVERAGINGCONFIG.fields_by_name['average_last_n_trees'])
_AVERAGINGCONFIG.fields_by_name['average_last_n_trees'].containing_oneof = _AVERAGINGCONFIG.oneofs_by_name['config']
_AVERAGINGCONFIG.oneofs_by_name['config'].fields.append(
  _AVERAGINGCONFIG.fields_by_name['average_last_percent_trees'])
_AVERAGINGCONFIG.fields_by_name['average_last_percent_trees'].containing_oneof = _AVERAGINGCONFIG.oneofs_by_name['config']
_LEARNERCONFIG.fields_by_name['regularization'].message_type = _TREEREGULARIZATIONCONFIG
_LEARNERCONFIG.fields_by_name['constraints'].message_type = _TREECONSTRAINTSCONFIG
_LEARNERCONFIG.fields_by_name['pruning_mode'].enum_type = _LEARNERCONFIG_PRUNINGMODE
_LEARNERCONFIG.fields_by_name['growing_mode'].enum_type = _LEARNERCONFIG_GROWINGMODE
_LEARNERCONFIG.fields_by_name['learning_rate_tuner'].message_type = _LEARNINGRATECONFIG
_LEARNERCONFIG.fields_by_name['multi_class_strategy'].enum_type = _LEARNERCONFIG_MULTICLASSSTRATEGY
_LEARNERCONFIG.fields_by_name['averaging_config'].message_type = _AVERAGINGCONFIG
_LEARNERCONFIG.fields_by_name['weak_learner_type'].enum_type = _LEARNERCONFIG_WEAKLEARNERTYPE
_LEARNERCONFIG.fields_by_name['each_tree_start'].message_type = tensorflow_dot_contrib_dot_boosted__trees_dot_proto_dot_tree__config__pb2._DECISIONTREECONFIG
_LEARNERCONFIG_PRUNINGMODE.containing_type = _LEARNERCONFIG
_LEARNERCONFIG_GROWINGMODE.containing_type = _LEARNERCONFIG
_LEARNERCONFIG_MULTICLASSSTRATEGY.containing_type = _LEARNERCONFIG
_LEARNERCONFIG_WEAKLEARNERTYPE.containing_type = _LEARNERCONFIG
_LEARNERCONFIG.oneofs_by_name['feature_fraction'].fields.append(
  _LEARNERCONFIG.fields_by_name['feature_fraction_per_tree'])
_LEARNERCONFIG.fields_by_name['feature_fraction_per_tree'].containing_oneof = _LEARNERCONFIG.oneofs_by_name['feature_fraction']
_LEARNERCONFIG.oneofs_by_name['feature_fraction'].fields.append(
  _LEARNERCONFIG.fields_by_name['feature_fraction_per_level'])
_LEARNERCONFIG.fields_by_name['feature_fraction_per_level'].containing_oneof = _LEARNERCONFIG.oneofs_by_name['feature_fraction']
DESCRIPTOR.message_types_by_name['TreeRegularizationConfig'] = _TREEREGULARIZATIONCONFIG
DESCRIPTOR.message_types_by_name['TreeConstraintsConfig'] = _TREECONSTRAINTSCONFIG
DESCRIPTOR.message_types_by_name['LearningRateConfig'] = _LEARNINGRATECONFIG
DESCRIPTOR.message_types_by_name['LearningRateFixedConfig'] = _LEARNINGRATEFIXEDCONFIG
DESCRIPTOR.message_types_by_name['LearningRateLineSearchConfig'] = _LEARNINGRATELINESEARCHCONFIG
DESCRIPTOR.message_types_by_name['AveragingConfig'] = _AVERAGINGCONFIG
DESCRIPTOR.message_types_by_name['LearningRateDropoutDrivenConfig'] = _LEARNINGRATEDROPOUTDRIVENCONFIG
DESCRIPTOR.message_types_by_name['LearnerConfig'] = _LEARNERCONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TreeRegularizationConfig = _reflection.GeneratedProtocolMessageType('TreeRegularizationConfig', (_message.Message,), {
  'DESCRIPTOR' : _TREEREGULARIZATIONCONFIG,
  '__module__' : 'tensorflow.contrib.boosted_trees.proto.learner_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.boosted_trees.learner.TreeRegularizationConfig)
  })
_sym_db.RegisterMessage(TreeRegularizationConfig)

TreeConstraintsConfig = _reflection.GeneratedProtocolMessageType('TreeConstraintsConfig', (_message.Message,), {
  'DESCRIPTOR' : _TREECONSTRAINTSCONFIG,
  '__module__' : 'tensorflow.contrib.boosted_trees.proto.learner_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.boosted_trees.learner.TreeConstraintsConfig)
  })
_sym_db.RegisterMessage(TreeConstraintsConfig)

LearningRateConfig = _reflection.GeneratedProtocolMessageType('LearningRateConfig', (_message.Message,), {
  'DESCRIPTOR' : _LEARNINGRATECONFIG,
  '__module__' : 'tensorflow.contrib.boosted_trees.proto.learner_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.boosted_trees.learner.LearningRateConfig)
  })
_sym_db.RegisterMessage(LearningRateConfig)

LearningRateFixedConfig = _reflection.GeneratedProtocolMessageType('LearningRateFixedConfig', (_message.Message,), {
  'DESCRIPTOR' : _LEARNINGRATEFIXEDCONFIG,
  '__module__' : 'tensorflow.contrib.boosted_trees.proto.learner_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.boosted_trees.learner.LearningRateFixedConfig)
  })
_sym_db.RegisterMessage(LearningRateFixedConfig)

LearningRateLineSearchConfig = _reflection.GeneratedProtocolMessageType('LearningRateLineSearchConfig', (_message.Message,), {
  'DESCRIPTOR' : _LEARNINGRATELINESEARCHCONFIG,
  '__module__' : 'tensorflow.contrib.boosted_trees.proto.learner_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.boosted_trees.learner.LearningRateLineSearchConfig)
  })
_sym_db.RegisterMessage(LearningRateLineSearchConfig)

AveragingConfig = _reflection.GeneratedProtocolMessageType('AveragingConfig', (_message.Message,), {
  'DESCRIPTOR' : _AVERAGINGCONFIG,
  '__module__' : 'tensorflow.contrib.boosted_trees.proto.learner_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.boosted_trees.learner.AveragingConfig)
  })
_sym_db.RegisterMessage(AveragingConfig)

LearningRateDropoutDrivenConfig = _reflection.GeneratedProtocolMessageType('LearningRateDropoutDrivenConfig', (_message.Message,), {
  'DESCRIPTOR' : _LEARNINGRATEDROPOUTDRIVENCONFIG,
  '__module__' : 'tensorflow.contrib.boosted_trees.proto.learner_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.boosted_trees.learner.LearningRateDropoutDrivenConfig)
  })
_sym_db.RegisterMessage(LearningRateDropoutDrivenConfig)

LearnerConfig = _reflection.GeneratedProtocolMessageType('LearnerConfig', (_message.Message,), {
  'DESCRIPTOR' : _LEARNERCONFIG,
  '__module__' : 'tensorflow.contrib.boosted_trees.proto.learner_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.boosted_trees.learner.LearnerConfig)
  })
_sym_db.RegisterMessage(LearnerConfig)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
