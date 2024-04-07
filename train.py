import argparse
import os
from glob import glob
import pandas as pd
import numpy as np
import random
import yaml
from utils import str2bool
from collections import OrderedDict
from sklearn.model_selection import train_test_split

from trainer import trainer
from dataset import CustomDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
import metrics
from metrics import Dice, IOU

from networks import networks
from networks.TransUNet.vit_configs import VIT_CONFIGS
from networks.TransUNet.TransUNet import TransUNet
from networks.NestedUNet.NestedUNet import NestedUNet


NETWORKS = networks.__all__
VIT_CONFIGS_LIST = list(VIT_CONFIGS.keys()) 
LOSSES = metrics.__all__

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Training pipeline
    parser.add_argument('--name', default=None, 
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=15, type=int, metavar='N',
                        help='number of epochs for training')
    parser.add_argument('--batch_size', default=3, type=int, metavar='N',
                        help='mini-batch size')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    
    # Network
    parser.add_argument('--type', default='Transformer', choices=['Transformer', 'Nested', 'Normal'],
                        help='type of networks: ' + ' | '.join(['Transformer', 'Nested', 'Normal'])
                        + 'default (Transformer)')
    parser.add_argument('--vit_name', default='R50-ViT-B_16',
                        help='vision transformer name:' 
                        + ' | '.join(VIT_CONFIGS_LIST) + '(default: R50-ViT-B_16)')
    parser.add_argument('--deep_supervision', default=False, help='deep supervision')
    parser.add_argument('--network', default='TransUnet', choices=NETWORKS,
                        help='networks: ' + ' | '.join(NETWORKS) 
                        + 'default: TransUnet')
    parser.add_argument('--input_channels', default=1, type=int,
                        help='input channels')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='input patch size')
    parser.add_argument('--num_classes', default=8, type=int,
                        help='number of classes')
    parser.add_argument('--width', default=512, type=int, 
                        help='input image width')
    parser.add_argument('--height', default=512, type=int,
                        help='input image height')
    
    # Criterion
    parser.add_argument('--loss', default='Dice cross entropy', choices=LOSSES)
    
    # Dataset
    parser.add_argument('--dataset', default='imagechd', help='dataset name')
    parser.add_argument('--ext', default='.npy', help='file extension')
    
    # Optimizer
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                        help='optimizer: ' + ' | '.join(['Adam', 'SGD']) 
                        + 'default (Adam)')
    parser.add_argument('--base_lr', '--learning_rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=0.0001, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    
    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 
                                 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    
    return parser.parse_args()
 
def output_config(config):
    print('-' * 20)
    for key in config:
        print(f'{key}: {config[key]}')
    print('-' * 20)   
    
def train():
    # Process the config
    config = parse_args()
    config_dict = vars(config)
    print(config.network)
    
    ## Specify model name
    config.name = f"{config.dataset}_{config.network}_bs{config.batch_size}_ps{config.patch_size}"
    if config.type == 'Transformer':
        config.name += f"_{config.vit_name}"
    elif config.type == 'Nested':
        if config['deep_supervision']:
            config.name += '_wDS'
        else:
            config.name += '_woDS'
    config.name += f"_epo{config.epochs}_hw{config.width}"
    
    if config.type != 'Transformer':
        config.vit_name = None
        config_dict['vit_name'] = None
        
    output_config(config_dict)
    
    ## Save config
    print(f"=> Initialize output: {config.name}")
    model_path = f"outputs/{config.name}"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        with open(f"{model_path}/config.yml", "w") as f:
            yaml.dump(config_dict, f)
    
    # Data Loading
    image_paths = glob(f"data/{config.dataset}/p_images/*{config.ext}")
    label_paths = glob(f"data/{config.dataset}/p_labels/*{config.ext}")
    
    train_image_paths, val_image_paths, train_label_paths, val_label_paths = train_test_split(image_paths, label_paths, test_size=0.2, random_state=41)
    train_ds = CustomDataset(train_image_paths, train_label_paths)
    val_ds = CustomDataset(val_image_paths, val_label_paths)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True)
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False)
    
    # Logging
    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        
        ('train_ce_loss', []),
        ('train_dice_loss', []),
        ('train_dice_score', []),
        ('train_dice_class_wise', []),
        ('train_iou', []),
        ('train_dice_ce_loss', []),
        
        ('val_ce_loss', []),
        ('val_dice_loss', []),
        ('val_dice_score', []),
        ('val_dice_class_wise', []),
        ('val_iou', []),
        ('val_dice_ce_loss', []),
    ])
    
    # Model
    print(f"=> Initialize model: {config.network}")
    if config.network == 'TransUnet':
        vit_config = VIT_CONFIGS[config.vit_name]
        vit_config.n_classes = config.num_classes
        vit_config.n_skip = 3
        
        if config.vit_name.find('R50') != -1:
            vit_config.patches.grid = (int(config.width / config.patch_size), int(config.width / config.patch_size))
            print(vit_config.patches.grid)
        
        print(f'=> Intialize vision transformer config: {vit_config}')
        model = TransUNet(config=vit_config, img_size=config.width, num_classes=config.num_classes).cuda()
    
    elif config.network == 'NestedUNet':
        model = NestedUNet(num_classes=config.num_classes, input_channels=1, deep_supervision=config.deep_supervision).cuda()

    else: raise "WRONG NETWORK NAME"
    
    # Optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.base_lr, weight_decay=config.weight_decay)
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config.base_lr, momentum=config.momentum,
        nesterov=config.nesterov, weight_decay=config.weight_decay)
        
    # Criterion
    ce = CrossEntropyLoss()
    dice = Dice(config.num_classes)
    
    # Training loop
    best_iou = 0
    best_dice_score = 0
    for epoch in range(config.epochs):
        print(f"Epoch: {epoch}/{config.epochs}")
        train_log = trainer(config, train_loader, optimizer, model, ce, dice)
        return 

    
if __name__ == '__main__':
    train()
    