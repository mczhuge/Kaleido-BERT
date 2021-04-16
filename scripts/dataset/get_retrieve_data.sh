#!/usr/bin/env sh
# This script downloads the pre-processed Fashion-gen retrieve datasets.

RT_DATA_DIR=retrieve
#DIR="$( cd "$(dirname "$0")" ; pwd -P )"

mkdir $RT_DATA_DIR

echo "=========== Prepare pre-processed Fashion-gen retrieve datasets ==========="
echo "Downloading..."

cd $RT_DATA_DIR

# i2t
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/retrieve/retrieve_i2t__01b600c2a5874bbfaea0bc89d855b771
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/retrieve/retrieve_i2t__52dccf651b204ae8bbe9f8e2e1b0a077
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/retrieve/retrieve_i2t__8bc6875d40134459b4a7da967545949c
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/retrieve/retrieve_i2t__a08201b321d1406fb4d4d8641ad62e1e
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/retrieve/retrieve_i2t__cdff430117d64b399b469ff89cc193ba

#t2i
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/retrieve/retrieve_t2i__05718d09802440d3b19ea87cb92eb7cc
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/retrieve/retrieve_t2i__2779a2e0cec14254a2bd0492665e4083
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/retrieve/retrieve_t2i__48cf4224d0a14e1abc3fe9e4aaf0a204
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/retrieve/retrieve_t2i__5e589686d0bc4d5ba5e43e6b95346532
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/retrieve/retrieve_t2i__c31c57c04c4e43f0b1019762591060fd

echo "Done..."
