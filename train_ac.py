import os
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
from tqdm import tqdm
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

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 2300000000

import warnings
warnings.filterwarnings('ignore')



def cleanup():
    dist.destroy_process_group()   


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')

parser.add_argument('--imgs_train', type=str, default='/dataset/dzy/danbooru2021/px640')
parser.add_argument('--imgs_val', type=str, default='/dataset/dzy/danbooru2021/px640')
parser.add_argument('--label_train', type=str, default='/data3/dzy/datas/danbooru2021/danbooru2021/data_train.json')
parser.add_argument('--label_val', type=str, default='/data3/dzy/datas/danbooru2021/danbooru2021/data_val.json')
parser.add_argument('--arb', type=str, default=None)
parser.add_argument('--out_dir', type=str, default='./models')

parser.add_argument('--adam_8bit', action="store_true", default=False)
parser.add_argument('--log_step', type=int, default=5)
parser.add_argument('--save_step', type=int, default=5000)
parser.add_argument('--ema_step', default=1, type=int)
parser.add_argument('--log_dir', type=str, default='logs/')

parser.add_argument('--default_ckpt', default=None, type=str)
parser.add_argument('--ema_ckpt', default=None, type=str)
parser.add_argument('--start_saving_epoch', default=3, type=str)


parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--start_step', default=0, type=int)

parser.add_argument('--epochs', default=15, type=int)
parser.add_argument('--ema', default=0.997, type=float)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--model_name', default='tresnet_l')
parser.add_argument('--model_path', default=None, type=str)
parser.add_argument('--num_classes', default=20943)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--image_size', default=448, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('--batch_size', default=10, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
parser.add_argument('--max_grad_norm', default=1.0, type=float)
parser.add_argument('--metadata_path', default='s3://pixai-test-uw2/richard/final_Danbooru_4.parquet', type=str)
parser.add_argument('--tag_index_dict_path', default='s3://pixai-test-uw2/richard/tags_index_dict.parquet', type=str)


# ML-Decoder
parser.add_argument('--use_ml_decoder', default=1, type=int)
parser.add_argument('--num_of_groups', default=512, type=int)  # full-decoding
parser.add_argument('--decoder_embedding', default=1024, type=int)
parser.add_argument('--zsl', default=0, type=int)
parser.add_argument('--num_layers_decoder', default=1, type=int)

parser.add_argument('--frelu', type=str2bool, default=True)
parser.add_argument('--xformers', type=str2bool, default=False)
parser.add_argument('--learn_query', type=str2bool, default=False)




class Trainer:
    def __init__(self, args):
        self.args=args

        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision='fp16',
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        )

        args.device = self.accelerator.device

        if self.accelerator.is_local_main_process:
            os.makedirs(args.log_dir, exist_ok=True)
            logger.add(os.path.join(args.log_dir, f'{time.strftime("%Y-%m-%d-%H-%M-%S")}.log'))

        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        self.world_size = self.accelerator.num_processes
        print('world_size: ', self.world_size)

        logger.info(f'rank: {self.local_rank}')
        if self.accelerator.is_local_main_process:
            logger.info(f'world_size: {self.world_size}')
            logger.info(f'accumulation: {self.accelerator.gradient_accumulation_steps}')

            os.makedirs(self.args.out_dir, exist_ok=True)

        self.make_lr()

        set_seed(41 + self.local_rank)

        self.build_model()
        self.build_data()
        self.build_optimizer_scheduler()


        self.model, self.optimizer, self.train_dataloader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )

        self.weight_dtype = torch.float32

        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16

    def make_lr(self):
        self.args.lr = (self.args.batch_size / 56) * self.args.lr * self.world_size * self.accelerator.gradient_accumulation_steps

        # Function to calculate the number of parameters
    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def build_model(self):
        # Setup model
        if self.accelerator.is_local_main_process:
            logger.info('creating model {}...'.format(self.args.model_name))
        self.model = create_model(self.args).cuda()
        print(f"Number of trainable parameters: {self.count_parameters(self.model)}")        

        if self.accelerator.is_local_main_process:
            logger.info('done')
            logger.info(f'lr_max: {self.args.lr}')

        self.ema = ModelEma(self.model, self.args.ema)  # 0.9997^641=0.82

        # load ckpt
        if self.args.default_ckpt:
            self.model.load_state_dict(torch.load(self.args.default_ckpt), strict = True)
        if self.args.ema_ckpt:
            self.ema.load_state_dict(torch.load(self.args.ema_ckpt), strict = True)
        


    def build_data(self):
        # Define a transformation pipeline that first resizes and then pads the images
        transform = transforms.Compose([
            ResizeAndPad(self.args.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

             
        train_dataset, val_dataset = self.initialize_train_val_split(transform)
        
        #print('train_dataset: ', type(train_dataset))
        
        if self.args.arb:
            train_dataset.make_arb(self.args.arb, self.args.batch_size*self.world_size)

        logger.info(f"len(val_dataset)): {len(val_dataset)}")
        logger.info(f"len(train_dataset)): {len(train_dataset)}")

                
        # Pytorch Data loader
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=self.world_size,
                                                                        rank=self.local_rank, shuffle=not self.args.arb)
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.args.batch_size,
            num_workers=self.args.workers, sampler=train_sampler)

        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=self.world_size,
                                                                      rank=self.local_rank, shuffle=False)
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.args.batch_size,
            num_workers=self.args.workers, sampler=val_sampler)

    def initialize_train_val_split(self, transform):
        df = pd.read_parquet(self.args.metadata_path)

        total_size = len(df)
        train_size = int(total_size * 0.000001)  # 80% for training
        val_size = int(total_size * 0.1)

        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size : train_size + val_size]
        
        
        train_dataset = S3Danbooru(train_df, self.args.tag_index_dict_path, self.args.num_classes, transform=transform)
        val_dataset = S3Danbooru(val_df, self.args.tag_index_dict_path, self.args.num_classes, transform=transform)

        return train_dataset, val_dataset
        

    def get_parameter_group(self):
        return add_weight_decay(self.model, self.args.weight_decay)

    def build_optimizer_scheduler(self):
        # set optimizer
        lr = self.args.lr
        self.criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
        parameters = self.get_parameter_group()

        if self.args.adam_8bit:
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(params=parameters, lr=lr, weight_decay=0)
        elif self.accelerator.state.deepspeed_plugin is not None:
            from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
            self.optimizer = FusedAdam(params=parameters, lr=lr, weight_decay=0)
        else:
            self.optimizer = torch.optim.AdamW(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn

        self.steps_per_epoch = len(self.train_loader)
        self.build_scheduler()

    def build_scheduler(self):
        self.scheduler = lr_scheduler.OneCycleLR(self.optimizer, max_lr=[x['lr'] for x in self.optimizer.state_dict()['param_groups']],
                                                 steps_per_epoch=self.steps_per_epoch, epochs=self.args.epochs, pct_start=0.2)

    def train(self):
        highest_mAP = 0
        loss_sum = 0

        self.start_step = self.args.start_step
        self.scheduler.step(self.args.start_epoch * self.steps_per_epoch + self.start_step)
        #print('self.train_loader.dataset: ', type(self.train_loader.dataset))
        self.train_loader.dataset.set_skip_imgs(self.start_step * self.args.batch_size * self.world_size)

        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.epoch=epoch
            if self.args.arb:
                self.train_loader.dataset.rest_arb(epoch)
            self.model.train()
            for i, (inputData, target) in tqdm(enumerate(self.train_loader)):
                #print('inputData: ', inputData.shape)
                #print('target: ', target.shape)
                if self.start_step > 0:
                    if i>=self.start_step-1:
                        self.start_step = -1
                        self.train_loader.dataset.set_skip_imgs(0)
                    continue

                loss = self.train_one_step(inputData, target, i)

                if loss is None:
                    break

                loss_sum+=loss
                # store information
                if self.accelerator.is_local_main_process:
                    if (i + 1) % self.args.log_step == 0:
                            logger.info('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                                        .format(self.epoch + 1, self.args.epochs, str(i + 1).zfill(3),
                                                str(self.steps_per_epoch).zfill(3),
                                                self.scheduler.get_last_lr()[0],
                                                loss_sum / self.args.log_step))
                            self.log_train_hook(i, self.args.log_step)
                            loss_sum = 0

                    if (i + 1) % self.args.save_step == 0:
                        self.save_model(self.args.model_name, i)

            if self.accelerator.is_local_main_process:
                self.save_model(self.args.model_name, i)

        self.model.eval()
        mAP_score = self.validate_multi(self.model, self.ema)

        if self.local_rank in [-1, 0]:
            if mAP_score > highest_mAP:
                print('Higher')
                highest_mAP = mAP_score
                self.save_model(self.args.model_name, i)
            logger.info('current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(mAP_score, highest_mAP))

            torch.cuda.synchronize()

    def log_train_hook(self, step, log_step):
        pass

    def train_one_step(self, inputData, target, step):
        with self.accelerator.accumulate(self.model):
            inputData = inputData.to(self.accelerator.device, dtype=self.weight_dtype)
            target = target.to(self.accelerator.device, dtype=int)

            loss, out=self.cal_loss(inputData, target)

            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            for x in out:
                del x
            del out
            del inputData
            del target

            if step % self.args.ema_step ==0:
                self.ema.update(self.model)

            #if self.start_step + step >= self.steps_per_epoch:
            #    self.start_step = -1
            #    return None
        return loss

    def cal_loss(self, inputData, target):
        output = self.model(inputData)  # sigmoid will be done in loss !
        loss = self.criterion(output, target)
        return loss, (output,)

    def save_model(self, model_name, step):
        if self.local_rank == 0:
            try:
                #torch.save({'model': self.model.state_dict(), 'ema': self.ema.state_dict()}, os.path.join(
                #    self.args.out_dir, f'{model_name}-{self.epoch + 1}-{step + 1}.ckpt'))
                
                torch.save(self.model, os.path.join(
                    self.args.out_dir, f"./default_{model_name}-{self.epoch + 1}-{step + 1}.pth")
                )
                
                torch.save(self.ema, os.path.join(
                    self.args.out_dir, f"./ema_{model_name}-{self.epoch + 1}-{step + 1}.pth")
                )


            except:
                pass
        else:
            print(f'Too early to save: epoch {self.epoch + 1}')

    def forward_val(self, model, input):
        #print(torch.sigmoid(model(input)[0]).shape)
        #return torch.sigmoid(model(input)[0]['pred_logits'])
        #print('input', input.shape)                    
        check = torch.sigmoid(model(input))
        #print('check: ', check.shape)
        
        return check
        
    def validate_multi(self, model, ema_model):
        logger.info("starting validation")
        preds_regular = []
        preds_ema = []
        targets = []
        with torch.no_grad():
            with autocast():
                for i, (input, target) in tqdm(enumerate(self.val_loader)):
                    input = input.to(self.accelerator.device, dtype=self.weight_dtype)
                    target = target.to(self.accelerator.device, dtype=int)
                    #print('target', target.shape)    


                    # compute output
                    output_regular = self.forward_val(model, input).cpu()
                    #print('output_regular: ', output_regular.shape)
                    output_ema = self.forward_val(ema_model, input).cpu()
                    #print('output_ema: ', output_ema.shape)

                    # for mAP calculation
                    preds_regular.append(output_regular.cpu())
                    preds_ema.append(output_ema.cpu())
                    targets.append(target.cpu())
        
        #print('preds_regular: ', len(preds_regular), preds_regular[0].shape, preds_regular[1].shape,)
        #print('targets: ', len(targets), targets[0].shape, targets[1].shape,)
        
        targets_cat = torch.cat(targets)
        preds_regular_cat = torch.cat(preds_regular)
        preds_ema_cat = torch.cat(preds_ema)
        
        #print('targets_cat: ', targets_cat.shape)
        #print('preds_regular_cat: ', preds_regular_cat.shape)
        
        # Local rank is correctly set to 0
        print('targets_cat: ', targets_cat.shape)
        print('preds_regular_cat: ', preds_regular_cat.shape)
        print('preds_ema_cat: ', preds_ema_cat.shape)
        
        
        
        if self.local_rank > -1:
            targets_all = Adist.gather(targets_cat, dst=0)
            print('targets_all: ', targets_all)
            preds_regular_all = Adist.gather(preds_regular_cat, dst=0)
            print('preds_regular_all: ', preds_regular_all)           
            preds_ema_all = Adist.gather(preds_ema_cat, dst=0)
            print('preds_ema_all: ', preds_ema_all)
            
        print('Here1')
        
        #print('targets_all: ', targets_all[0].shape)
        #print('preds_regular_all: ', preds_regular_all.shape)

        #print('targs: ', torch.cat(targets_all).numpy().shape)
        #print('preds: ', torch.cat(preds_regular_all).numpy().shape)
        if self.local_rank in [-1, 0]:
            mAP_score_regular = mAP(torch.cat(targets_all).numpy(), torch.cat(preds_regular_all).numpy())
            print('Here2')
            
            mAP_score_ema = mAP(torch.cat(targets_all).numpy(), torch.cat(preds_ema_all).numpy())            
            logger.info("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
            return max(mAP_score_regular, mAP_score_ema)
        else:
            return 0



def make_fake_args():
    # !python train_ac.py --imgs_train ./data --imgs_val ./data --label_train ./fake_labels.json --label_val ./fake_labels.json
    args = parser.parse_args([])
    args.imgs_train = './data'
    args.imgs_val = './data'
    args.label_train = './fake_labels.json'
    args.label_val = './fake_labels.json'
    args.batch_size = 4
    args.xformers = False
    return args


if __name__ == '__main__':

    args = parser.parse_args()
    #args = make_fake_args()
    trainer =Trainer(args)
    trainer.train()