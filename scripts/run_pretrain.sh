#!/usr/bin/env bash
# coding=utf-8
# Copyright (c) 2021 Alibaba Group. Licensed under the MIT license.

echo "================ Kaleido-BERT Pretraining ==============="

ls -d ./dataset/pretrain/pretrain_train* > ./dataset/train_list.list_csv
ls -d ./dataset/pretrain/pretrain_valid* > ./dataset/dev_list.list_csv

export CUDA_VISIBLE_DEVICES="0,1,2,3"

python pretrain_main.py \
  --workerGPU=4 \
  --mode=train_and_evaluate \
  --train_input_fp=./dataset/train_list.list_csv  \
  --eval_input_fp=./dataset/dev_list.list_csv  \
  --pretrain_model_name_or_path=pai-kaleidobert-base-en  \
  --input_sequence_length=64  \
  --train_batch_size=64  \
  --num_epochs=25  \
  --model_dir=./checkpoint/pretrained  \
  --learning_rate=1e-4  \
  --image_feature_size=131072  \
  --input_schema="input_ids:int:64,input_mask:int:64,segment_ids:int:64,masked_lm_positions:int:10,masked_lm_ids:int:10,masked_lm_weights:float:10,img_feature_convert_rotation:float:2048,img_feature_convert_jigsaw:float:8192,img_feature_convert_camouflage:float:18432,img_feature_convert_grey_mask:float:32768,img_feature_convert_blank_mask:float:51200,image_mask:int:55,img_loc_position_rotation:int:5,img_loc_position_jigsaw:int:20,img_loc_position_camouflage:int:35,img_loc_position_grey_mask:int:80,img_loc_position_blank_mask:int:125,img_subtasks_flag:int:1,img_grey_lm_ids:int:2,img_blank_lm_ids:int:3,img_rotate_gt:int:1,img_jigsaw_gt:int:1,img_camouflage_gt:int:1,img_grey_mask_gt:float:4096,nx_sent_labels:int:1"  \

