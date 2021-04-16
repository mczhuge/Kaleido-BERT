#!/usr/bin/env bash
# coding=utf-8
# Copyright (c) 2021 Alibaba Group. Licensed under the MIT license.

task=subcatepred

echo "================ Kaleido-BERT ${task} Finetune ==============="
ls -d ./datasets/finetune/finetune_train_* > ./datasets/train_${task}_list.list_csv
ls -d ./datasets/finetune/finetune_valid_* > ./datasets/eval_${task}_list.list_csv

export CUDA_VISIBLE_DEVICES="4,5,6,7"

python finetune_main.py \
  --workerGPU=4 \
  --mode=train_and_evaluate \
  --task=${task} \
  --train_input_fp=./datasets/train_${task}_list.list_csv  \
  --eval_input_fp=./datasets/eval_${task}_list.list_csv  \
  --pretrain_model_name_or_path=pai-kaleidobert-base-en  \
  --input_sequence_length=64  \
  --train_batch_size=64  \
  --num_epochs=20  \
  --model_dir=./checkpoint/classification/${task}  \
  --learning_rate=2e-5  \
  --output_dir=./checkpoint/classification/${task}_out  \
  --predict_checkpoint_path=./checkpoint/pretrained/kaleidobert.ckpt-50683  \
  --input_schema="text_prod_id:str:1,input_ids:int:64,input_mask:int:64,segment_ids:int:64,prod_desc:str:1,image_prod_id:str:1,prod_img_id:str:1,img_feature_convert_rotation:float:2048,img_feature_convert_jigsaw:float:8192,img_feature_convert_camouflage:float:18432,img_feature_convert_grey_mask:float:32768,img_feature_convert_blank_mask:float:51200,image_mask:int:55,img_loc_position_rotation:int:5,img_loc_position_jigsaw:int:20,img_loc_position_camouflage:int:35,img_loc_position_grey_mask:int:80,img_loc_position_blank_mask:int:125,img_category:str:1,img_category_id:int:1,img_subcategory:str:1,img_subcategory_id:int:1"  \


