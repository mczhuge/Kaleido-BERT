#!/usr/bin/env sh
# This script downloads the pre-processed Fashion-gen retrieve datasets.

RAW_DATA_DIR=raw
#DIR="$( cd "$(dirname "$0")" ; pwd -P )"

mkdir $RAW_DATA_DIR

echo "=========== Download raw Fashion-gen datsets ==========="
echo "Downloading..."

cd $RAW_DATA_DIR

#GET RAW DATA
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/raw_data/extracted_images.tar.gz
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/raw_data/full_train_info.txt
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/raw_data/full_valid_info.txt

#SOME ROI SAMPLES
#wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/raw_data/roi_demo/

#Pytorch tensor dataset
#wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/torch_dataset/datasets/fashiongen_pretrain_train_feats_1.pt
#wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/torch_dataset/datasets/fashiongen_pretrain_train_feats_2.pt
#wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/torch_dataset/datasets/fashiongen_pretrain_train_feats_3.pt
#wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/torch_dataset/datasets/fashiongen_pretrain_train_feats_4.pt
#wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/torch_dataset/datasets/fashiongen_pretrain_train_feats_5.pt
#wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/torch_dataset/datasets/tsv_to_pt_1000.py
#wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/torch_dataset/datasets/tsv_to_pt_all_split.py



#Kaleido Patch Generator
#wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/torch_dataset/preprocess/KaleidoBERT-patch-1106-train.py
#wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/torch_dataset/preprocess/KaleidoBERT-patch-1106-valid.py 

echo "Done..."
