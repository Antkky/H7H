# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: response_rithmic_system_info.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor # type: ignore
from google.protobuf import message as _message # type: ignore
from google.protobuf import reflection as _reflection # type: ignore
from google.protobuf import symbol_database as _symbol_database # type: ignore
from google.protobuf import descriptor_pb2 # type: ignore
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='response_rithmic_system_info.proto',
  package='rti',
  serialized_pb=_b('\n\"response_rithmic_system_info.proto\x12\x03rti\"\x91\x01\n\x19ResponseRithmicSystemInfo\x12\x15\n\x0btemplate_id\x18\xe3\xb6\t \x02(\x05\x12\x12\n\x08user_msg\x18\x98\x8d\x08 \x03(\t\x12\x11\n\x07rp_code\x18\x9e\x8d\x08 \x03(\t\x12\x15\n\x0bsystem_name\x18\x9c\xb0\t \x03(\t\x12\x1f\n\x15has_aggregated_quotes\x18\xb1\xb0\t \x03(\x08')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_RESPONSERITHMICSYSTEMINFO = _descriptor.Descriptor(
  name='ResponseRithmicSystemInfo',
  full_name='rti.ResponseRithmicSystemInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='template_id', full_name='rti.ResponseRithmicSystemInfo.template_id', index=0,
      number=154467, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='user_msg', full_name='rti.ResponseRithmicSystemInfo.user_msg', index=1,
      number=132760, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='rp_code', full_name='rti.ResponseRithmicSystemInfo.rp_code', index=2,
      number=132766, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='system_name', full_name='rti.ResponseRithmicSystemInfo.system_name', index=3,
      number=153628, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='has_aggregated_quotes', full_name='rti.ResponseRithmicSystemInfo.has_aggregated_quotes', index=4,
      number=153649, type=8, cpp_type=7, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=44,
  serialized_end=189,
)

DESCRIPTOR.message_types_by_name['ResponseRithmicSystemInfo'] = _RESPONSERITHMICSYSTEMINFO

ResponseRithmicSystemInfo = _reflection.GeneratedProtocolMessageType('ResponseRithmicSystemInfo', (_message.Message,), dict(
  DESCRIPTOR = _RESPONSERITHMICSYSTEMINFO,
  __module__ = 'response_rithmic_system_info_pb2'
  # @@protoc_insertion_point(class_scope:rti.ResponseRithmicSystemInfo)
  ))
_sym_db.RegisterMessage(ResponseRithmicSystemInfo)


# @@protoc_insertion_point(module_scope)
