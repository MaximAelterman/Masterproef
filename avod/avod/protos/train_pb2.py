# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: avod/protos/train.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from avod.protos import optimizer_pb2 as avod_dot_protos_dot_optimizer__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='avod/protos/train.proto',
  package='avod.protos',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n\x17\x61vod/protos/train.proto\x12\x0b\x61vod.protos\x1a\x1b\x61vod/protos/optimizer.proto\"\xef\x02\n\x0bTrainConfig\x12\x15\n\nbatch_size\x18\x01 \x01(\r:\x01\x31\x12\x1b\n\x0emax_iterations\x18\x02 \x02(\r:\x03\x35\x30\x30\x12)\n\toptimizer\x18\x03 \x01(\x0b\x32\x16.avod.protos.Optimizer\x12\x1f\n\x13\x63heckpoint_interval\x18\x04 \x01(\r:\x02\x35\x30\x12#\n\x17max_checkpoints_to_keep\x18\x05 \x01(\r:\x02\x31\x30\x12$\n\x15overwrite_checkpoints\x18\x06 \x01(\x08:\x05\x66\x61lse\x12\x1c\n\x10summary_interval\x18\x07 \x02(\r:\x02\x31\x30\x12\x1a\n\x12summary_histograms\x18\x08 \x02(\x08\x12\x1a\n\x12summary_img_images\x18\t \x02(\x08\x12\x1a\n\x12summary_bev_images\x18\n \x02(\x08\x12#\n\x14\x61llow_gpu_mem_growth\x18\x0b \x01(\x08:\x05\x66\x61lse')
  ,
  dependencies=[avod_dot_protos_dot_optimizer__pb2.DESCRIPTOR,])




_TRAINCONFIG = _descriptor.Descriptor(
  name='TrainConfig',
  full_name='avod.protos.TrainConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='batch_size', full_name='avod.protos.TrainConfig.batch_size', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_iterations', full_name='avod.protos.TrainConfig.max_iterations', index=1,
      number=2, type=13, cpp_type=3, label=2,
      has_default_value=True, default_value=500,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='optimizer', full_name='avod.protos.TrainConfig.optimizer', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='checkpoint_interval', full_name='avod.protos.TrainConfig.checkpoint_interval', index=3,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=50,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_checkpoints_to_keep', full_name='avod.protos.TrainConfig.max_checkpoints_to_keep', index=4,
      number=5, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=10,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='overwrite_checkpoints', full_name='avod.protos.TrainConfig.overwrite_checkpoints', index=5,
      number=6, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='summary_interval', full_name='avod.protos.TrainConfig.summary_interval', index=6,
      number=7, type=13, cpp_type=3, label=2,
      has_default_value=True, default_value=10,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='summary_histograms', full_name='avod.protos.TrainConfig.summary_histograms', index=7,
      number=8, type=8, cpp_type=7, label=2,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='summary_img_images', full_name='avod.protos.TrainConfig.summary_img_images', index=8,
      number=9, type=8, cpp_type=7, label=2,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='summary_bev_images', full_name='avod.protos.TrainConfig.summary_bev_images', index=9,
      number=10, type=8, cpp_type=7, label=2,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='allow_gpu_mem_growth', full_name='avod.protos.TrainConfig.allow_gpu_mem_growth', index=10,
      number=11, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
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
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=70,
  serialized_end=437,
)

_TRAINCONFIG.fields_by_name['optimizer'].message_type = avod_dot_protos_dot_optimizer__pb2._OPTIMIZER
DESCRIPTOR.message_types_by_name['TrainConfig'] = _TRAINCONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TrainConfig = _reflection.GeneratedProtocolMessageType('TrainConfig', (_message.Message,), dict(
  DESCRIPTOR = _TRAINCONFIG,
  __module__ = 'avod.protos.train_pb2'
  # @@protoc_insertion_point(class_scope:avod.protos.TrainConfig)
  ))
_sym_db.RegisterMessage(TrainConfig)


# @@protoc_insertion_point(module_scope)
