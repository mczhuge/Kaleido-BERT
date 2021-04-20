#!/usr/bin/env sh
# This script downloads the pre-processed Fashion-gen pretrain datasets.

PT_DATA_DIR=pretrain
#DIR="$F cd "$(dirname "$0")" ; pwd -P )"

mkdir $PT_DATA_DIR
echo "=========== Prepare pre-processed Fashion-gen pre-training datasets ==========="
echo "Downloading..."
cd $PT_DATA_DIR

#train
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_train__007b71adc2cd4faaa341892805dd71c4
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_train__0175b257c2ad4b72b03109d28a38b8e3
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_train__0846293575774213aeb85e1654430b7a
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_train__0ce3dbe3260b47df81bf63fe805342f0
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_train__0ef31b341fa74217a3addcf621dc58f2
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_train__1ed75d4f75524a1da7084b6b2e38ccb9
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_train__26cff0c4b56b407ba0bf77e4669315e0
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_train__2e7750470f4d4f7c8e9de43bfb488835
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_train__3d247259becc4fdaa5c550e7373921ec
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_train__5cf573d247ba4768921a8cb1dad13796
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_train__5f19270a66a246b0853afc0866c45bd5
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_train__6de21b9e18c54744ae926c3e782f942f
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_train__760e509454f64290949f2bc6b1193738
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_train__7687cde4548e423cb455efea17a16799
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_train__7cb7ba8fa28444a6b46d7d131127a35f
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_train__8704393f3ffc479f85d4755d31d2afff
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_train__8b5476882ea6472391cc781340925d73
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_train__9331e4a63db44ee38512779c7aa274cd
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_train__a5142a0e48b04ee181b310c0628b3aa6
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_train__afd85c233ff0420b805024fc93a9350a
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_train__bbc1e27fe2e847d8a53be4518d540004
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_train__bfad0cff538b435ea945f64917cc0b6e
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_train__c1b9e938fdd542fdb71b73c517342c12
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_train__c4dfa3ed9bad4118bc60775032387621
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_train__c61457fe3ac84376991f5bff4b6cef65
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_train__cb6d6f5c6b154e299f5335dd76890f65
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_train__d55995fb494f45d98412182add27884a
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_train__de1c0914ac0d4715a41fc326c0e583a1
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_train__f41adfb8165b4bd594d853c06087c236
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_train__ff3569ea813b404c983c05da05afc7e7

#valid
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_valid__35c3fbcf16684842b4a78dfa6d98bc69
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_valid__4717b37fde4649fe81a3a07b8f275225
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_valid__72f50fe93e8549e18b1adb895869b6ff
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_valid__7696fe0e58a74227920574114b5eaebf
wget http://icbu-ensa-sc.oss-cn-zhangjiakou.aliyuncs.com/mingchen.zgmc/KaleidoBERT_TF_CODE/datasets/pretrain/pretrain_valid__f979043f2b1a415b8965cf9a6471183c

echo "Done!"




 




















