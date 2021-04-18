#!/usr/bin/env sh
# This script downloads the Kaleido-BERT pretrained checkpoint and EasyTransfer Dependancy.

PRE_CKPT=pretrained
EZ_DEP=~/.eztransfer_modelzoo
#DIR="$( cd "$(dirname "$0")" ; pwd -P )"

mkdir $PRE_CKPT

echo "=========== Prepare pretrained checkpoint ==========="
echo "Downloading..."

cd $PRE_CKPT
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/checkpoint/config.json
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/checkpoint/kaleidobert.ckpt-50683.data-00000-of-00001
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/checkpoint/kaleidobert.ckpt-50683.index
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/checkpoint/kaleidobert.ckpt-50683.meta

echo "=========== Prepare EasyTransfer Dependency ==========="
mkdir $EZ_DEP
cd $EZ_DEP
echo "Downloading..."
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/eztransfer_modelzoo/pai-kaleidobert-base-en.zip
unzip pai-kaleidobert-base-en.zip
 
echo "Done..."
