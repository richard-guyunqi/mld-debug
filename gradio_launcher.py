# Need to modify data into tensor, not address

import os
import argparse
import time

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms

from src_files.data.path_dataset import PathDataset
from src_files.helper_functions.helper_functions import crop_fix
from src_files.models import create_model
from tqdm.auto import tqdm

import pandas as pd
import json
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]


class Demo:
    def __init__(self, args):
        self.args=args
        print('args in demo.py: ', args)

        print('creating model {}...'.format(args.model_name))
        args.model_path = None
        model = create_model(args, load_head=True).to(device)
        state = torch.load(args.ckpt, map_location='cpu')
        if args.ema:
            state = state['ema']
        elif 'model' in state:
            state=state['model']
        model.load_state_dict(state, strict=True)

        self.model = model.to(device).eval()
        #######################################################
        print('done')

        if args.keep_ratio:
            self.trans = transforms.Compose([
                transforms.Resize(args.image_size),
                crop_fix,
                transforms.ToTensor(),
            ])
        else:
            self.trans = transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                ])

        self.load_class_map()

    def load_class_map(self):
        self.class_map = self.generate_dict(pd.read_parquet(self.args.tag_index_dict_path))        
        #print(self.class_map.keys())
            
    def generate_dict(self, df):
        tags_to_index_dict = {idx: tag for idx, tag in enumerate(df['Tags'].unique())}
        return tags_to_index_dict
    
    def load_data(self, img):
        img = self.trans(img)
        return img

    def infer_one(self, img):
        with torch.cuda.amp.autocast(enabled=self.args.fp16):
            img = img.unsqueeze(0)
            output = torch.sigmoid(self.model(img)).cpu().view(-1)
        pred = torch.where(output > self.args.thr)[0].numpy()

        cls_list = [(self.class_map[i], output[i]) for i in pred]
        return cls_list

#     @torch.no_grad()
#     def infer(self, path):
#         if os.path.isfile(path):
#             img = self.load_data(path).to(device)
#             cls_list = self.infer_one(img)
#             return cls_list
#         else:
#             tag_dict = {}
#             img_list = [os.path.join(path, x) for x in os.listdir(path) if x[x.rfind('.'):].lower() in IMAGE_EXTENSIONS]
#             for item in tqdm(img_list):
#                 img = self.load_data(item).to(device)
#                 cls_list = self.infer_one(img)
#                 cls_list.sort(reverse=True, key=lambda x: x[1])
#                 if self.args.out_type == 'txt':
#                     with open(item[:item.rfind('.')] + '.txt', 'w', encoding='utf8') as f:
#                         f.write(', '.join([name.replace('_', ' ') for name, prob in cls_list]))
#                 elif self.args.out_type == 'json':
#                     tag_dict[os.path.basename(item)] = ', '.join([name.replace('_', ' ') for name, prob in cls_list])

#             if self.args.out_type == 'json':
#                 with open(os.path.join(path, 'image_captions.json'), 'w', encoding='utf8') as f:
#                     f.write(json.dumps(tag_dict, indent=2, ensure_ascii=False))

#             return None

    @torch.no_grad()
    def infer(self, path):
        img = self.load_data(path).to(device)
        cls_list = self.infer_one(img)
        return cls_list
       
