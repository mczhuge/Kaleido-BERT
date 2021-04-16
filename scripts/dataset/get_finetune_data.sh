#!/usr/bin/env sh
# This script downloads the pre-processed Fashion-gen fine-tune(cls) datasets.

RT_DATA_DIR=finetune
#DIR="$F cd "$(dirname "$0")" ; pwd -P )"

mkdir $FT_DATA_DIR
echo "=========== Prepare pre-processed Fashion-gen classification datasets ==========="
echo "Downloading..."

cd $FT_DATA_DIR

# finetune train
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/finetune/finetune_train__05746b2d5ce143a6be1b464b77845b34
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/finetune/finetune_train__0b1a26bfe3704d619d187ffbb4e58868
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/finetune/finetune_train__658eb9240fbc406b883d46f64c1cc832
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/finetune/finetune_train__c1745b8e0fd84715aea9426a821df51b
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/finetune/finetune_train__e976961bf490411ba49252a72afb2428

#finetune valid
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/finetune/finetune_valid__24f777078c1c44e49b35f7cb24dbb203
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/finetune/finetune_valid__2e31d958c6eb42c791df3070e721f56f
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/finetune/finetune_valid__49bc9638fee04ab7bbe2a448f5611575
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/finetune/finetune_valid__862543bc6a5243208081412f6a3c95eb
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/finetune/finetune_valid__bbd6a0358e0f454b878fa8eeae08266d

echo "Done..."
