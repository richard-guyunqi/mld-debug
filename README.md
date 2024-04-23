# Files explanation:
## In active Use:
demo.ca.py: Inference for caformer-mld
train_ca_acpy: Training file for the our model, basically a wrapper for train_ac.py
train_ac: The main body of training code, originally for tresnet
gradio_demo_launcher.py: Gradio demo starter code
test_dataset: A small demo dataset, 20 images, both common and uncommon images
test.ipynb: Some random notebook used to run py files and check environment
sematics_test.ipynb: Some random notebook used to run py files and check environment

## Path to Danbooru metadata
s3://pixai-test-uw2/richard/final_Danbooru_5.parquet

## Path to model checkpoints
s3://pixai-test-uw2/richard/mld-ckpts/records/

## Path to tag-index dictionary
s3://pixai-test-uw2/richard/tags_index_dict.parquet


## Not in active use:
demo.py: Inference for tresnet-mld (not used)
class.json



# ML-Danbooru: Anime image tags detector

## Introduction
An anime image tag detector based on modified [ML-Decoder](https://github.com/Alibaba-MIIL/ML_Decoder).
Model trained with cleaned [danbooru2021](https://gwern.net/danbooru2021).

+ Designed a new TResNet-D structure as backbone to enhance the learning of low-level features.
+ Replace the ReLU in backbone with [FReLU](https://arxiv.org/pdf/2007.11824.pdf).
+ Using learnable queries for transformer decoder.

## Model Structure

![](./imgs/ml_danbooru.png)

## Model-Zoo
https://huggingface.co/7eu7d7/ML-Danbooru

## Usage
Download the model and run below command:
```bash
python demo.py --data <path to image or directory> --model_name tresnet_d --num_of_groups 32 --ckpt <path to ckpt> --thr 0.7 --image_size 640 
```

Keep the image ratio invariant:
```bash
python demo.py --data <path to image or directory> --model_name tresnet_d --num_of_groups 32 --ckpt <path to ckpt> --thr 0.7 --image_size 640 --keep_ratio True
```

### ML_CAFormer
```bash
python demo_ca.py --data <path to image or directory> --model_name caformer_m36 --ckpt <path to ckpt> --thr 0.7 --image_size 448
```