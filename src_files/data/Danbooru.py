import torch
from torch.utils.data import Dataset
import json
from PIL import Image
from PIL import ImageFile
import cv2
import boto3
import pandas as pd
import os
import numpy as np
from copy import deepcopy, copy
from torchvision import transforms
from torchvision.transforms.functional import pad
from PIL import Image
from io import BytesIO


ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 2300000000

class Danbooru(Dataset):
    def __init__(self, root, annFile, num_class=53109, transform=None, file_ext=None):
        self.root = root
        self.transform = transform
        self.num_class = num_class
        self.file_ext = file_ext

        with open(annFile, 'r', encoding='utf8') as f:
            self.labels = json.loads(f.read())
        self.data_len = len(self.labels)

        self.skip_num=0
        self.arb=None

    def make_arb(self, arb=None, bs=None):
        self.arb = arb
        self.bs = bs
        rs = np.random.RandomState(42)

        self.idx_imgid_map={item[0]:i for i,item in enumerate(self.labels)}

        with open(arb, 'r', encoding='utf8') as f:
            self.bucket_dict = json.loads(f.read())

        # make len(bucket)%bs==0
        self.data_len = 0
        self.bucket_list=[]
        for k,v in self.bucket_dict.items():
            bucket=copy(v)
            rest=len(v)%bs
            if rest>0:
                bucket.extend(rs.choice(v, bs-rest))
            self.data_len += len(bucket)
            self.bucket_list.append(np.array(bucket))

    def rest_arb(self, epoch):
        rs = np.random.RandomState(42+epoch)
        bucket_list = [copy(x) for x in self.bucket_list]
        #shuffle inter bucket
        for x in bucket_list:
            rs.shuffle(x)

        # shuffle of batches
        bucket_list=np.hstack(bucket_list).reshape(-1, self.bs)
        rs.shuffle(bucket_list)

        self.labels_arb = bucket_list.reshape(-1)

    def __getitem__(self, index):
        if index<self.skip_num:
            return torch.Tensor(0), torch.Tensor(0)

        if self.arb:
            item = self.labels[self.idx_imgid_map[self.labels_arb[index]]] # index -> imgid -> label_idx
        else:
            item = self.labels[index]

        img_path = self.get_path(item)
        img = self.get_image(img_path)

        target = torch.zeros(self.num_class, dtype=torch.float)
        target[item[2]]=1.

        return img, target

    def set_skip_imgs(self, skip_num):
        self.skip_num=skip_num

    def get_path(self, item):
        return os.path.join(self.root, f'{item[0]}.{self.file_ext if self.file_ext else item[1]}')

    def get_image(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
        
        except PIL.UnidentifiedImageError:
            print('Error at image_path: ', image_path)
            print('Df row num: ', i)
            
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return self.data_len

class S3Danbooru(Dataset):
    def __init__(self, df, tag_index_dict_path, num_class=20586, transform=None):
        """
        Args:
            metadata_path (string): Path to the .parquet file with image S3 paths and labels.
            num_class (int): Number of classes.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df
        self.transform = transform
        self.num_class = num_class
        self.s3 = boto3.client('s3')
        self.dict = self.generate_dict(pd.read_parquet(tag_index_dict_path))
        self.data_len = len(self.df)

        self.skip_num=0
        self.arb=None

    def __len__(self):
        return self.data_len
    
    def parse_s3_path(self, s3_path):
        """
        Parse the S3 path to get the bucket name and key.
        """
        if s3_path.startswith("s3://"):
            s3_path = s3_path[5:]
        bucket, key = s3_path.split("/", 1)
        return bucket, key 
    
    def parse_labels(self, raw_labels):
        return raw_labels.split('|')
    
    def generate_dict(self, df):
        tags_to_index_dict = {tag: idx for idx, tag in enumerate(df['Tags'].unique())}
        return tags_to_index_dict
        
    
    def text_tag_to_index(self, tag):
        index = self.dict.at[tag, 'Index']  # Using `.at` for scalar access
        return index
    
    
    def __getitem__(self, idx):
        # Correctly access the row for the given idx
        row = self.df.iloc[idx]
        
        s3_path = row['s3_uri']
        raw_labels = row['merged_tags']
        labels = self.parse_labels(raw_labels)
        bucket, key = self.parse_s3_path(s3_path)

        # Fetch the image from S3
        response = self.s3.get_object(Bucket=bucket, Key=key)
        image = Image.open(BytesIO(response['Body'].read())).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Convert label to a tensor
        target = torch.zeros(self.num_class, dtype=torch.float)
        for label in labels:
            # Make sure label is an integer or convert it to an integer index
            try:
                label_idx = int(self.dict[label])  # This line might need adjustment based on your label format
                target[label_idx] = 1
            except KeyError:
                pass
        return image, target
    
    
    def make_arb(self, arb=None, bs=None):
        self.arb = arb
        self.bs = bs
        rs = np.random.RandomState(42)

        self.idx_imgid_map={item[0]:i for i,item in enumerate(self.labels)}

        with open(arb, 'r', encoding='utf8') as f:
            self.bucket_dict = json.loads(f.read())

        # make len(bucket)%bs==0
        self.data_len = 0
        self.bucket_list=[]
        for k,v in self.bucket_dict.items():
            bucket=copy(v)
            rest=len(v)%bs
            if rest>0:
                bucket.extend(rs.choice(v, bs-rest))
            self.data_len += len(bucket)
            self.bucket_list.append(np.array(bucket))

    def rest_arb(self, epoch):
        rs = np.random.RandomState(42+epoch)
        bucket_list = [copy(x) for x in self.bucket_list]
        #shuffle inter bucket
        for x in bucket_list:
            rs.shuffle(x)

        # shuffle of batches
        bucket_list=np.hstack(bucket_list).reshape(-1, self.bs)
        rs.shuffle(bucket_list)

        self.labels_arb = bucket_list.reshape(-1)
        
    def set_skip_imgs(self, skip_num):
        self.skip_num=skip_num

        

class ResizeAndPad:
    def __init__(self, target_size, fill=0, padding_mode='constant'):
        self.target_size = target_size
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        # Calculate the scale to resize the image so the long side is self.target_size
        scale = self.target_size / max(img.width, img.height)
        new_width = int(img.width * scale)
        new_height = int(img.height * scale)
        
        # Resize the image
        img = transforms.Resize((new_height, new_width))(img)
        
        # Calculate padding to make the image square (self.target_size x self.target_size)
        pad_left = (self.target_size - new_width) // 2
        pad_top = (self.target_size - new_height) // 2
        pad_right = self.target_size - new_width - pad_left
        pad_bottom = self.target_size - new_height - pad_top
        
        # Apply padding
        img = pad(img, (pad_left, pad_top, pad_right, pad_bottom), self.fill, self.padding_mode)
        return img
