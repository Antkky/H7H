# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: request_login_info.proto

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
  name='request_login_info.proto',
  package='rti',
  serialized_pb=_b('\n\x18request_login_info.proto\x12\x03rti\"=\n\x10RequestLoginInfo\x12\x15\n\x0btemplate_id\x18\xe3\xb6\t \x02(\x05\x12\x12\n\x08user_msg\x18\x98\x8d\x08 \x03(\t')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_REQUESTLOGININFO = _descriptor.Descriptor(
  name='RequestLoginInfo',
  full_name='rti.RequestLoginInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='template_id', full_name='rti.RequestLoginInfo.template_id', index=0,
      number=154467, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='user_msg', full_name='rti.RequestLoginInfo.user_msg', index=1,
      number=132760, type=9, cpp_type=9, label=3,
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
  serialized_start=33,
  serialized_end=94,
)

DESCRIPTOR.message_types_by_name['RequestLoginInfo'] = _REQUESTLOGININFO

RequestLoginInfo = _reflection.GeneratedProtocolMessageType('RequestLoginInfo', (_message.Message,), dict(
  DESCRIPTOR = _REQUESTLOGININFO,
  __module__ = 'request_login_info_pb2'
  # @@protoc_insertion_point(class_scope:rti.RequestLoginInfo)
  ))
_sym_db.RegisterMessage(RequestLoginInfo)


# @@protoc_insertion_point(module_scope)
