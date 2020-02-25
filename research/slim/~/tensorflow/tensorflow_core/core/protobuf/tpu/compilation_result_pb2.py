# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/core/protobuf/tpu/compilation_result.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from tensorflow.compiler.xla.service import hlo_pb2 as tensorflow_dot_compiler_dot_xla_dot_service_dot_hlo__pb2
from tensorflow.core.lib.core import error_codes_pb2 as tensorflow_dot_core_dot_lib_dot_core_dot_error__codes__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensorflow/core/protobuf/tpu/compilation_result.proto',
  package='tensorflow.tpu',
  syntax='proto3',
  serialized_options=_b('\370\001\001'),
  serialized_pb=_b('\n5tensorflow/core/protobuf/tpu/compilation_result.proto\x12\x0etensorflow.tpu\x1a)tensorflow/compiler/xla/service/hlo.proto\x1a*tensorflow/core/lib/core/error_codes.proto\"\x86\x01\n\x16\x43ompilationResultProto\x12+\n\x0bstatus_code\x18\x01 \x01(\x0e\x32\x16.tensorflow.error.Code\x12\x1c\n\x14status_error_message\x18\x02 \x01(\t\x12!\n\nhlo_protos\x18\x03 \x03(\x0b\x32\r.xla.HloProtoB\x03\xf8\x01\x01\x62\x06proto3')
  ,
  dependencies=[tensorflow_dot_compiler_dot_xla_dot_service_dot_hlo__pb2.DESCRIPTOR,tensorflow_dot_core_dot_lib_dot_core_dot_error__codes__pb2.DESCRIPTOR,])




_COMPILATIONRESULTPROTO = _descriptor.Descriptor(
  name='CompilationResultProto',
  full_name='tensorflow.tpu.CompilationResultProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='status_code', full_name='tensorflow.tpu.CompilationResultProto.status_code', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='status_error_message', full_name='tensorflow.tpu.CompilationResultProto.status_error_message', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='hlo_protos', full_name='tensorflow.tpu.CompilationResultProto.hlo_protos', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=161,
  serialized_end=295,
)

_COMPILATIONRESULTPROTO.fields_by_name['status_code'].enum_type = tensorflow_dot_core_dot_lib_dot_core_dot_error__codes__pb2._CODE
_COMPILATIONRESULTPROTO.fields_by_name['hlo_protos'].message_type = tensorflow_dot_compiler_dot_xla_dot_service_dot_hlo__pb2._HLOPROTO
DESCRIPTOR.message_types_by_name['CompilationResultProto'] = _COMPILATIONRESULTPROTO
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

CompilationResultProto = _reflection.GeneratedProtocolMessageType('CompilationResultProto', (_message.Message,), {
  'DESCRIPTOR' : _COMPILATIONRESULTPROTO,
  '__module__' : 'tensorflow.core.protobuf.tpu.compilation_result_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.tpu.CompilationResultProto)
  })
_sym_db.RegisterMessage(CompilationResultProto)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
