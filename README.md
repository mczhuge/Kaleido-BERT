## [Kaleido-BERT: Vision-Language Pre-training on Fashion Domain](https://arxiv.org/pdf/2101.07663.pdf)
Mingchen Zhuge*, Dehong Gao*, Deng-Ping Fan#, Linbo Jin, Ben Chen, Haoming Zhou, Minghui Qiu, Ling Shao.

[**[Paper]**](https://arxiv.org/pdf/2101.07663.pdf)

## Updates
1) Code will be released in 2021/4/16.
2) This is the tensorflow implementation built on [Alibaba/EasyTransfer](https://github.com/alibaba/EasyTransfer). 
   We will also release a Pytorch version built on [Huggingface/Transformers](https://github.com/huggingface/transformers) in future.
   
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
cd Kaleido-BERT/scipts/dataset
sh get_finetune_dataset.sh
sh get_retrieve_dataset.sh

#Testing CAT/SUB

cd Kaleido-BERT/scipts
sh run_sub.sh
sh run_subcat.sh

#

cd Kaleido-BERT/scipts
sh run_i2t.sh
sh run_t2i.sh
```
5. Pre-training
```
rm -rf Kaleido-BERT/checkpoint/pretrained
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
 

                 
   Feel free to contact me if you have additional questions. 
