import os
import argparse
import time

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from src_files.helper_functions.helper_functions import mAP, CutoutPIL, ModelEma, \
    add_weight_decay
from src_files.helper_functions.bn_fusion import fuse_bn_recursively
from src_files.models import create_model
from tqdm.auto import tqdm
import json
import argparse

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from src_files.helper_functions.helper_functions import mAP, CutoutPIL, ModelEma, \
    add_weight_decay
from src_files.data.Danbooru import Danbooru, S3Danbooru, ResizeAndPad
from src_files.data.utils import ResizeArea, WeakRandAugment
from src_files.models import create_model
from src_files.loss_functions.losses import AsymmetricLoss
from torch.cuda.amp import GradScaler, autocast

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from src_files import dist as Adist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
import numpy as np
import random
from loguru import logger
import time
import pandas as pd

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate import DistributedDataParallelKwargs

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from torchvision import transforms
from torchvision.transforms.functional import pad
from PIL import Image
from torch.utils.data import random_split

import torch
from torch.utils.data import Dataset
import json
from PIL import Image
from PIL import ImageFile
import cv2
import os
import numpy as np
from copy import deepcopy, copy
from io import BytesIO

use_abn=True
try:
    from src_files.models.tresnet.tresnet import InplacABN_to_ABN
except:
    use_abn=False

import json
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def make_args():
    parser = argparse.ArgumentParser(description='PyTorch MS_COCO validation')
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--class_map', type=str, default='./class.json')
    parser.add_argument('--model_name', default='tresnet_l')
    parser.add_argument('--num_classes', default=20943)
    parser.add_argument('--image_size', default=448, type=int,
                        metavar='N', help='input image size')
    parser.add_argument('--thr', default=0.75, type=float,
                        metavar='N', help='threshold value')
    parser.add_argument('--keep_ratio', type=str2bool, default=False)
    
    parser.add_argument('--tag_index_dict_path', default='s3://pixai-test-uw2/richard/tags_index_dict.parquet', type=str)


    # ML-Decoder
    parser.add_argument('--use_ml_decoder', default=1, type=int)
    parser.add_argument('--num_of_groups', default=512, type=int)  # full-decoding
    parser.add_argument('--decoder_embedding', default=1024, type=int)
    parser.add_argument('--zsl', default=0, type=int)
    parser.add_argument('--fp16', action="store_true", default=False)
    
    # Ema(value) or nothing, indicating which kind of ckpt/model is being used
    parser.add_argument('--ema',default=None, type=float)

    parser.add_argument('--frelu', type=str2bool, default=True)
    parser.add_argument('--xformers', type=str2bool, default=False)

    parser.add_argument('--out_type', type=str, default='txt')

    args = parser.parse_args()
    return args


def crop_fix(img: Image):
    w, h = img.size
    w = (w // 4) * 4
    h = (h // 4) * 4
    return img.crop((0, 0, w, h))

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


class Demo:
    def __init__(self, args):
        self.args = args

        print('creating model {}...'.format(args.model_name))
        args.model_path = None
        model = create_model(args).to(device)
        
        #print(model)
        if self.args.ckpt:
            if not args.ema:
                print('default: ', self.args.ckpt)
                model.load_state_dict(torch.load(self.args.ckpt), strict=True)
            else:
                ema = ModelEma(model, self.args.ema)  # 0.9997^641=0.82
                print('ema: ', self.args.ckpt)           
                ema.load_state_dict(torch.load(self.args.ckpt), strict=True)
                model = ema
            
#         if not args.xformers and 'head.decoder.layers.0.multihead_attn.in_proj_container.q_proj.weight' in state:
#             in_proj_weight = torch.cat([state['head.decoder.layers.0.multihead_attn.in_proj_container.q_proj.weight'],
#                                         state['head.decoder.layers.0.multihead_attn.in_proj_container.k_proj.weight'],
#                                         state[
#                                             'head.decoder.layers.0.multihead_attn.in_proj_container.v_proj.weight'], ],
#                                        dim=0)
#             in_proj_bias = torch.cat([state['head.decoder.layers.0.multihead_attn.in_proj_container.q_proj.bias'],
#                                       state['head.decoder.layers.0.multihead_attn.in_proj_container.k_proj.bias'],
#                                       state['head.decoder.layers.0.multihead_attn.in_proj_container.v_proj.bias'], ],
#                                      dim=0)
#             state['head.decoder.layers.0.multihead_attn.out_proj.weight'] = state[
#                 'head.decoder.layers.0.multihead_attn.proj.weight']
#             state['head.decoder.layers.0.multihead_attn.out_proj.bias'] = state[
#                 'head.decoder.layers.0.multihead_attn.proj.bias']
#             state['head.decoder.layers.0.multihead_attn.in_proj_weight'] = in_proj_weight
#             state['head.decoder.layers.0.multihead_attn.in_proj_bias'] = in_proj_bias

#             del state['head.decoder.layers.0.multihead_attn.in_proj_container.q_proj.weight']
#             del state['head.decoder.layers.0.multihead_attn.in_proj_container.k_proj.weight']
#             del state['head.decoder.layers.0.multihead_attn.in_proj_container.v_proj.weight']
#             del state['head.decoder.layers.0.multihead_attn.in_proj_container.q_proj.bias']
#             del state['head.decoder.layers.0.multihead_attn.in_proj_container.k_proj.bias']
#             del state['head.decoder.layers.0.multihead_attn.in_proj_container.v_proj.bias']
#             del state['head.decoder.layers.0.multihead_attn.proj.weight']
#             del state['head.decoder.layers.0.multihead_attn.proj.bias']


        model.eval()
        ########### eliminate BN for faster inference ###########
        model = model.cpu()
        if use_abn:
            model = InplacABN_to_ABN(model)
        model = fuse_bn_recursively(model)
        self.model = model.to(device).eval()
        if args.fp16:
            self.model = self.model.half()
        #######################################################
        print('done')

        self.trans = transforms.Compose([
            ResizeAndPad(self.args.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.load_class_map()

    def load_class_map(self):
        self.class_map = self.generate_dict(pd.read_parquet(self.args.tag_index_dict_path))
        #print('self.class_map.keys()[0]": ', type(list(self.class_map.keys())[0]))
        #print('self.class_map.keys()": ', list(self.class_map.keys())[0])

    def generate_dict(self, df):
        index_to_tags_dict = {idx: tag for idx, tag in enumerate(df['Tags'].unique())}
        return index_to_tags_dict

    def load_data(self, path):
        img = Image.open(path).convert('RGB')
        img = self.trans(img)
        #print('img_dimension: ', img.shape)
        return img

    def infer_one(self, img):
        if self.args.fp16:
            img = img.half()
        img = img.unsqueeze(0)
        output = torch.sigmoid(self.model(img)).cpu().view(-1)
        pred = torch.where(output > self.args.thr)[0].numpy()

        cls_list = [(self.class_map[i], output[i]) for i in pred]
        return cls_list

    @torch.no_grad()
    def infer(self, path):
        if os.path.isfile(path):
            img = self.load_data(path).to(device)
            cls_list = self.infer_one(img)
            return cls_list
        else:
            tag_dict={}
            img_list=[os.path.join(path, x) for x in os.listdir(path) if x[x.rfind('.'):].lower() in IMAGE_EXTENSIONS]
            for item in tqdm(img_list):
                img = self.load_data(item).to(device)
                cls_list = self.infer_one(img)
                cls_list.sort(reverse=True, key=lambda x: x[1])
                if self.args.out_type=='txt':
                    with open(item[:item.rfind('.')]+'.txt', 'w', encoding='utf8') as f:
                        f.write(', '.join([name.replace('_', ' ') for name, prob in cls_list]))
                elif self.args.out_type=='json':
                    tag_dict[os.path.basename(item)]=', '.join([name.replace('_', ' ') for name, prob in cls_list])

            if self.args.out_type == 'json':
                with open(os.path.join(path, 'image_captions.json'), 'w', encoding='utf8') as f:
                    f.write(json.dumps(tag_dict))

            return None

if __name__ == '__main__':
    args = make_args()
    demo = Demo(args)
    cls_list = demo.infer(args.data)

    if cls_list is not None:
        cls_list.sort(reverse=True, key=lambda x: x[1])
        print(f'Results for {args.data}: \n' )
        print(', '.join([f'{name}:{prob:.3}' for name, prob in cls_list]))
        print(', '.join([name for name, prob in cls_list]))