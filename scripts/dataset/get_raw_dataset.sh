#!/usr/bin/env sh
# This script downloads the pre-processed Fashion-gen retrieve datasets.

RT_DATA_DIR=retrieve
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
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/raw_data/roi_demo/

echo "Done..."
