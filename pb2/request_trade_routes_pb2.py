# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: request_trade_routes.proto

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
  name='request_trade_routes.proto',
  package='rti',
  serialized_pb=_b('\n\x1arequest_trade_routes.proto\x12\x03rti\"`\n\x12RequestTradeRoutes\x12\x15\n\x0btemplate_id\x18\xe3\xb6\t \x02(\x05\x12\x12\n\x08user_msg\x18\x98\x8d\x08 \x03(\t\x12\x1f\n\x15subscribe_for_updates\x18\xf0\xb5\t \x01(\x08')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_REQUESTTRADEROUTES = _descriptor.Descriptor(
  name='RequestTradeRoutes',
  full_name='rti.RequestTradeRoutes',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='template_id', full_name='rti.RequestTradeRoutes.template_id', index=0,
      number=154467, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='user_msg', full_name='rti.RequestTradeRoutes.user_msg', index=1,
      number=132760, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='subscribe_for_updates', full_name='rti.RequestTradeRoutes.subscribe_for_updates', index=2,
      number=154352, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
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
  serialized_start=35,
  serialized_end=131,
)

DESCRIPTOR.message_types_by_name['RequestTradeRoutes'] = _REQUESTTRADEROUTES

RequestTradeRoutes = _reflection.GeneratedProtocolMessageType('RequestTradeRoutes', (_message.Message,), dict(
  DESCRIPTOR = _REQUESTTRADEROUTES,
  __module__ = 'request_trade_routes_pb2'
  # @@protoc_insertion_point(class_scope:rti.RequestTradeRoutes)
  ))
_sym_db.RegisterMessage(RequestTradeRoutes)


# @@protoc_insertion_point(module_scope)
