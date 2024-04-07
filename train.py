import argparse
import os
from glob import glob
import pandas as pd
import numpy as np
import random
import yaml
from utils import str2bool

from networks import networks
import losses

import ml_collections


NETWORKS = networks.__all__
LOSSES = losses.__all__

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Training pipeline
    parser.add_argument('--name', default=None, 
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=15, type=int, metavar='N',
                        help='number of epochs for training')
    parser.add_argument('--batch_size', default=16, type=int, metavar='N',
                        help='mini-batch size')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    
    # Network
    parser.add_argument('--type', default='Transformer', choices=['Transformer', 'Nested'],
                        help='type of networks: ' + ' | '.join(['Transformer', 'Nested'])
                        + 'default (Transformer)')
    parser.add_argument('--n_skip', type=int,
                        default=3, help='number of skip-connect, default (3)')
    parser.add_argument('--deep_supervision', default=False, help='deep supervision')
    parser.add_argument('--network', default='TransUnet', choices=NETWORKS,
                        help='networks: ' + ' | '.join(NETWORKS) 
                        + 'default: TransUnet')
    parser.add_argument('--input_channels', default=1, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=8, type=int,
                        help='number of classes')
    parser.add_argument('--width', default=256, type=int, 
                        help='input image width')
    parser.add_argument('--height', default=256, type=int,
                        help='input image height')
    
    # Criterion
    parser.add_argument('--loss', default='Dice cross entropy', choices=LOSSES)
    
    # Dataset
    parser.add_argument('--dataset', default='imagechd', help='dataset name')
    parser.add_argument('--ext', default='.npy', help='file extension')
    
    # Optimizer
    parser.add_argument('--optmizer', default='Adam', choices=['Adam', 'SGD'],
                        help='optimizer: ' + ' | '.join(['Adam', 'SGD']) 
                        + 'default (Adam)')
    parser.add_argument('--base_lr', '--learning_rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
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
    config = vars(parse_args())
    output_config(config)
    
    config['name'] = f'{config['dataset']}_{config['network']}_bs{config['batch_size']}'
    if config['type'] == 'Transformer':
        config['name'] += f'_{config['n_skip']}'
    elif config['type'] == 'Nested':
        if config['deep_supervision']:
            config['name'] += '_wDS'
        else:
            config['name'] += '_woDS'
    config['name'] += f'_{config['epochs']}_{config['width']}'

    print(config['name'])
            
    
    
if __name__ == '__main__':
    train()