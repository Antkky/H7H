# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: response_account_list.proto

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
  name='response_account_list.proto',
  package='rti',
  serialized_pb=_b('\n\x1bresponse_account_list.proto\x12\x03rti\"\xac\x02\n\x13ResponseAccountList\x12\x15\n\x0btemplate_id\x18\xe3\xb6\t \x02(\x05\x12\x12\n\x08user_msg\x18\x98\x8d\x08 \x03(\t\x12\x1c\n\x12rq_handler_rp_code\x18\x9c\x8d\x08 \x03(\t\x12\x11\n\x07rp_code\x18\x9e\x8d\x08 \x03(\t\x12\x10\n\x06\x66\x63m_id\x18\x9d\xb3\t \x01(\t\x12\x0f\n\x05ib_id\x18\x9e\xb3\t \x01(\t\x12\x14\n\naccount_id\x18\x98\xb3\t \x01(\t\x12\x16\n\x0c\x61\x63\x63ount_name\x18\x92\xb3\t \x01(\t\x12\x1a\n\x10\x61\x63\x63ount_currency\x18\x8f\xb6\t \x01(\t\x12 \n\x16\x61\x63\x63ount_auto_liquidate\x18\xdb\xff\x07 \x01(\t\x12*\n auto_liq_threshold_current_value\x18\xe0\xff\x07 \x01(\t')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_RESPONSEACCOUNTLIST = _descriptor.Descriptor(
  name='ResponseAccountList',
  full_name='rti.ResponseAccountList',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='template_id', full_name='rti.ResponseAccountList.template_id', index=0,
      number=154467, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='user_msg', full_name='rti.ResponseAccountList.user_msg', index=1,
      number=132760, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='rq_handler_rp_code', full_name='rti.ResponseAccountList.rq_handler_rp_code', index=2,
      number=132764, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='rp_code', full_name='rti.ResponseAccountList.rp_code', index=3,
      number=132766, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='fcm_id', full_name='rti.ResponseAccountList.fcm_id', index=4,
      number=154013, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='ib_id', full_name='rti.ResponseAccountList.ib_id', index=5,
      number=154014, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='account_id', full_name='rti.ResponseAccountList.account_id', index=6,
      number=154008, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='account_name', full_name='rti.ResponseAccountList.account_name', index=7,
      number=154002, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='account_currency', full_name='rti.ResponseAccountList.account_currency', index=8,
      number=154383, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='account_auto_liquidate', full_name='rti.ResponseAccountList.account_auto_liquidate', index=9,
      number=131035, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='auto_liq_threshold_current_value', full_name='rti.ResponseAccountList.auto_liq_threshold_current_value', index=10,
      number=131040, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
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
  serialized_start=37,
  serialized_end=337,
)

DESCRIPTOR.message_types_by_name['ResponseAccountList'] = _RESPONSEACCOUNTLIST

ResponseAccountList = _reflection.GeneratedProtocolMessageType('ResponseAccountList', (_message.Message,), dict(
  DESCRIPTOR = _RESPONSEACCOUNTLIST,
  __module__ = 'response_account_list_pb2'
  # @@protoc_insertion_point(class_scope:rti.ResponseAccountList)
  ))
_sym_db.RegisterMessage(ResponseAccountList)


# @@protoc_insertion_point(module_scope)
