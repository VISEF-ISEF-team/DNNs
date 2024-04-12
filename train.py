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

from trainer import trainer, validate, write_csv
from dataset import CustomDataset
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
import metrics
from metrics import Dice, IOU

from networks import networks
from networks.TransUNet.vit_configs import VIT_CONFIGS
from networks.TransUNet.TransUNet import TransUNet
from networks.NestedUNet.NestedUNet import NestedUNet
from networks.UNet.UNet import UNet
from networks.NestedUNet.NestedUNet import NestedUNet
from networks.UNetAtt.UNetAtt import UNetAtt
from networks.NestedResUNetAtt.NestedResUNetAtt import NestedResUNetAtt
from networks.ResUNet.ResUNet import ResUNet
from networks.RotCAttTransUNetDense.RotCAttTransUNetDense import RotCAttTransUNetDense
from networks.RotCAttTransUNetDense.configs import get_config

NETWORKS = networks.__all__
VIT_CONFIGS_LIST = list(VIT_CONFIGS.keys()) 
LOSSES = metrics.__all__

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Training pipeline
    parser.add_argument('--name', default=None, 
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=65, type=int, metavar='N',
                        help='number of epochs for training')
    parser.add_argument('--batch_size', default=12, type=int, metavar='N',
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
    parser.add_argument('--network', default='RotCAttTransUNetDense', choices=NETWORKS,
                        help='networks: ' + ' | '.join(NETWORKS) 
                        + 'default: RotCAttTransUNetDense')
    parser.add_argument('--input_channels', default=1, type=int,
                        help='input channels')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='input patch size')
    parser.add_argument('--num_classes', default=8, type=int,
                        help='number of classes')
    parser.add_argument('--width', default=128, type=int, 
                        help='input image width')
    parser.add_argument('--height', default=128, type=int,
                        help='input image height')
    
    # Criterion
    parser.add_argument('--loss', default='Dice iou cross entropy', choices=LOSSES)
    
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
    
def train(config):
    # Process the config
    config_dict = vars(config)
    print(config.network)
    
    ## Specify model name
    config.name = f"{config.dataset}_{config.network}_bs{config.batch_size}_ps{config.patch_size}"
    if config.type == 'Transformer' and config.network == 'TransUNet':
        config.name += f"_{config.vit_name}"
    elif config.type == 'Nested':
        if config.deep_supervision:
            config.name += '_wDS'
        else:
            config.name += '_woDS'
    config.name += f"_epo{config.epochs}_hw{config.width}"
    
    if config.type != 'Transformer' or config.network != 'TransUNet':
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
    image_paths = glob(f"data/{config.dataset}/p_images/*{config.ext}")[:1000]
    label_paths = glob(f"data/{config.dataset}/p_labels/*{config.ext}")[:1000]
    
    train_image_paths, val_image_paths, train_label_paths, val_label_paths = train_test_split(image_paths, label_paths, test_size=0.2, random_state=41)
    train_ds = CustomDataset(train_image_paths, train_label_paths, img_size=config.width)
    val_ds = CustomDataset(val_image_paths, val_label_paths, img_size=config.width)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False
    )
        
    # Loggingc
    log = OrderedDict([
        ('epoch', []),                                          # 0
        ('lr', []),                                             # 1
        
        ('Train loss', []),                                     # 2
        ('Train ce loss', []),                                  # 3
        ('Train dice score', []),                               # 4
        ('Train dice loss', []),                                # 5
        ('Train iou score', []),                                # 6
        ('Train iou loss', []),                                 # 7
        
        ('Val loss', []),                                       # 8
        ('Val ce loss', []),                                    # 9
        ('Val dice score', []),                                 # 10
        ('Val dice loss', []),                                  # 11
        ('Val iou score', []),                                  # 12
        ('Val iou loss', [])                                    # 13
    ])
    
    # Model
    print(f"=> Initialize model: {config.network}")
    if config.network == 'TransUnet':
        vit_config = VIT_CONFIGS[config.vit_name]
        vit_config.n_classes = config.num_classes
        vit_config.n_skip = 3
        
        if config.vit_name.find('R50') != -1:
            vit_config.patches.grid = (int(config.width / config.patch_size), int(config.width / config.patch_size))
        
        print(f'=> Intialize vision transformer config: {vit_config}')
        model = TransUNet(config=vit_config, img_size=config.width, num_classes=config.num_classes).cuda()
    
    elif config.network == 'NestedUNet':
        model = NestedUNet(num_classes=config.num_classes, input_channels=1, deep_supervision=config.deep_supervision).cuda()
    
    elif config.network == 'UNet':
        model = UNet(num_classes=config.num_classes).cuda()
        
    elif config.network == 'NestedUNet':
        model = NestedUNet(num_classes=config.num_classes).cuda()
    
    elif config.network == 'UNetAtt':
        model = UNetAtt(num_classes=config.num_classes).cuda()
    
    elif config.network == 'NestedResUNetAtt':
        model = NestedResUNetAtt(num_classes=config.num_classes).cuda()
    
    elif config.network == 'ResUNet':
        model = ResUNet(num_classes=config.num_classes).cuda()
        
    elif config.network == 'RotCAttTransUNetDense':
        model_config = get_config()
        model = RotCAttTransUNetDense(model_config).cuda()
        
        
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
    iou = IOU(config.num_classes)
    
    # Training loop
    best_iou = 0
    best_dice_score = 0

    fieldnames = ['CE Loss', 'Dice Score', 'Dice Loss', 'IoU Score', 'IoU Loss', 'Total Loss']
    write_csv(f'outputs/{config.name}/iter_log.csv', fieldnames)

    for epoch in range(config.epochs):
        print(f"Epoch: {epoch+1}/{config.epochs}")
        train_log = trainer(config, train_loader, optimizer, model, ce, dice, iou)
        val_log = validate(config, val_loader, model, ce, dice, iou)
        
        print(f"Train loss: {train_log['loss']} - Train ce loss: {train_log['ce_loss']} - Train dice score: {train_log['dice_score']} - Train dice loss: {train_log['dice_loss']} - Train iou Score: {train_log['iou_score']} - Train iou loss: {train_log['iou_loss']}")
        print(f"Val loss: {val_log['loss']} - Val ce loss: {val_log['ce_loss']} - Val dice score: {val_log['dice_score']} - Val dice loss: {val_log['dice_loss']} - Val iou Score: {val_log['iou_score']} - Val iou loss: {val_log['iou_loss']}")
        
        log['epoch'].append(epoch)
        log['lr'].append(config.base_lr)
        
        log['Train loss'].append(train_log['loss'])
        log['Train ce loss'].append(train_log['ce_loss'])
        log['Train dice score'].append(train_log['dice_score'])
        log['Train dice loss'].append(train_log['dice_loss'])
        log['Train iou score'].append(train_log['iou_score'])
        log['Train iou loss'].append(train_log['iou_loss']) 
        
        log['Val loss'].append(val_log['loss'])
        log['Val ce loss'].append(val_log['ce_loss'])
        log['Val dice score'].append(val_log['dice_score'])
        log['Val dice loss'].append(val_log['dice_loss'])
        log['Val iou score'].append(val_log['iou_score'])
        log['Val iou loss'].append(val_log['iou_loss']) 
        
        pd.DataFrame(log).to_csv(f'outputs/{config.name}/epo_log.csv', index=False)
        
        if train_log['iou_score'] > best_iou and train_log['dice_score'] > best_dice_score:
            best_iou = train_log['iou_score']
            best_dice_score = train_log['dice_score']
            torch.save(model, f"outputs/{config.name}/model.pth")
    
if __name__ == '__main__':
    config = parse_args()
    train(config)
    