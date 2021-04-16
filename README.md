## [Kaleido-BERT: Vision-Language Pre-training on Fashion Domain](https://arxiv.org/pdf/2101.07663.pdf)
Mingchen Zhuge*, Dehong Gao*, Deng-Ping Fan#, Linbo Jin, Ben Chen, Haoming Zhou, Minghui Qiu, Ling Shao.

[**[Paper]**](https://arxiv.org/pdf/2101.07663.pdf)[**[Video]**](https://arxiv.org/pdf/2101.07663.pdf)

## Introduction
We present a new vision-language (VL) pre-training model dubbed Kaleido-BERT, which introduces a novel kaleido strategy for fashion cross-modality representations from transformers. In contrast to random masking strategy of recent VL models,  we design alignment guided masking to jointly focus more on image-text semantic relations. 
To this end, we carry out five novel tasks, \ie, rotation, jigsaw, camouflage, grey-to-color, and blank-to-color for self-supervised VL pre-training at patches of different scale. Kaleido-BERT is conceptually simple and easy to extend to the existing BERT framework, it attains state-of-the-art results by large margins on four downstream tasks, including text retrieval (R@1: 4.03\% absolute improvement), image retrieval (R@1: 7.13\% abs imv.), category recognition (ACC: 3.28\% abs imv.), and fashion captioning (Bleu4: 1.2 abs imv.). We validate the efficiency of \ourmodel~on a wide range of e-commercial websites, demonstrating its broader potential in real-world applications.
![framework](model.png) 

## Noted
1) Code will be released in 2021/4/16.
2) This is the tensorflow implementation built on [Alibaba/EasyTransfer](https://github.com/alibaba/EasyTransfer). 
   We will also release a Pytorch version built on [Huggingface/Transformers](https://github.com/huggingface/transformers) in future.
3) If you feel hard to download these datasets, you can modify `/dataset/get_pretrain_data.sh`, `/dataset/get_pretrain_data.sh`, `/dataset/get_pretrain_data.sh`, and comment out some `wget #file_links` as you want. This will not 
   
## Get started
1. Clone this code
```
git clone git@github.com:mczhuge/Kaleido-BERT.git
cd Kaleido-BERT
```
2. Enviroment setup (Details can be found on conda_env.info)
```
conda create  --name kaleidobert --file conda_env.info
conda activate kaleidobert
conda install tensorflow==1.15.0
pip install boto3 tqdm tensorflow_datasets --index-url=https://mirrors.aliyun.com/pypi/simple/
pip install sentencepiece==0.1.92 sklearn --index-url=https://mirrors.aliyun.com/pypi/simple/
pip install joblib==0.14.1
python setup.py develop
```
3. Download Dependancy
```
cd Kaleido-BERT/scripts/checkpoint
sh get_checkpoint.sh
```
4. Finetune
```
#Download finetune datasets

cd Kaleido-BERT/scipts/dataset
sh get_finetune_dataset.sh
sh get_retrieve_dataset.sh

#Testing CAT/SUB

cd Kaleido-BERT/scipts
sh run_sub.sh
sh run_subcat.sh

#Testing TIR/ITR

cd Kaleido-BERT/scipts
sh run_i2t.sh
sh run_t2i.sh
```
5. Pre-training
```
#Download pre-training datasets

cd Kaleido-BERT/scipts/dataset
sh get_prtrain_dataset.sh

#Remove existed checkpoint
rm -rf Kaleido-BERT/checkpoint/pretrained

#Run pre-training
cd Kaleido-BERT/scipts/
sh run_pretrain.sh
```

## Acknowlegement
Thanks Alibaba ICBU Search Team and Alibaba PAI Team for technical support.

## Citing Kaleido-BERT
```
@inproceedings{Zhuge2021KaleidoBERT,
  title={Kaleido-BERT: Vision-Language Pre-training on Fashion Domain},
  author={Zhuge, Mingchen and Gao, Dehong and Fan, Deng-Ping and Jin, Linbo and Chen, Ben and Zhou, Haoming and Qiu, Minghui and Shao, Ling},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={},
  year={2021}
}
```

## Contact
* Mingchen Zhuge (email: mczhuge@cug.edu.cn | wechat: tjpxiaoming)

* Deng-Ping Fan (email: denpfan@gmail.com)

                 
Feel free to contact us if you have additional questions. 
