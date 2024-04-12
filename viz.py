import argparse
import yaml
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import torch
from scipy.ndimage import zoom
from metrics import Dice, IOU

# Training graphs
def visualize(epochs, scores, legends, x_label, y_label, title, config):
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan'] 
    for score, legend, color in zip(scores, legends, colors):
        plt.plot(epochs, score, color, label=legend)
        
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(f"outputs/{config['name']}/graph2.jpeg")
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None, help='model name')
    return parser.parse_args()

def plotting():
    args = parse_args()
    with open(f'outputs/{args.name}/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    df = pd.read_csv(f"outputs/{config['name']}/epo_log.csv")
    fields = df.columns.tolist()  
    metrics = []
    for column in df.columns:
        metrics.append(df[column].tolist())
        
    iters = [i for i in range(1, (len(metrics[0])) + 1)]
    visualize(iters, [metrics[4], metrics[2], metrics[10], metrics[8]],  [fields[4], fields[2], fields[10], fields[8]],
            'Epochs', 'Scores', 'Training results', config)
    

# Prediction comparision
def compare(gt, label):
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))  
    axes[0].set_title('Ground truth')
    axes[0].imshow(gt) 
    axes[0].axis('off')
    
    axes[1].set_title('Label')
    axes[1].imshow(label) 
    axes[1].axis('off') 
    
    plt.tight_layout() 
    plt.show()
    
def rot_val():
    model = torch.load('outputs/imagechd_ResUNet_bs12_ps16_epo65_hw128/model.pth')
    vol = []
    labels = []
    for index in range(120, 132, 1):
        input = np.load(f'data/imagechd/p_images/0001_0{index}.npy')
        label = np.load(f'data/imagechd/p_labels/0001_0{index}.npy')
        
        x,y = input.shape
        img_size = 128
        input = zoom(input, (img_size / x, img_size / y), order=0)
        label = zoom(label, (img_size / x, img_size / y), order=0)
        
        input = torch.tensor(input).unsqueeze(0)
        
        vol.append(input)
        labels.append(label)
        
    vol = torch.stack(vol, dim=0).cuda()
    logits = model(vol)
    
    for index in range(12):
        val, pred = torch.max(logits, dim=1)
        pred = pred[0].detach().cpu().numpy()
        label = labels[index]
        compare(pred, label)
    
def val_pipeline():
    # Input
    model = torch.load('outputs/imagechd_ResUNet_bs12_ps16_epo65_hw128/model.pth')
    input = np.load('data/imagechd/p_images/0001_0120.npy')
    label = np.load('data/imagechd/p_labels/0001_0120.npy')
    
    # Dataloader
    x,y = input.shape
    img_size = 192
    input = zoom(input, (img_size / x, img_size / y), order=0)
    label = zoom(label, (img_size / x, img_size / y), order=0)
    
    # Prediction
    input = torch.tensor(input).unsqueeze(0).unsqueeze(0).cuda()
    logits = model(input)
    val, pred = torch.max(logits, dim=1)
    
    # detach
    pred = pred[0].detach().cpu().numpy()
    
    # Visualize
    viz_label = zoom(label, (x / img_size, y / img_size), order=0)
    viz_pred = zoom(pred, (x / img_size, y / img_size), order=0)
    compare(
        viz_label,
        viz_pred
    )
    
    # Dice
    label = torch.tensor(label).unsqueeze(0).cuda()
    dl = Dice(num_classes=8)
    ds, dl, class_ds, class_dl = dl(logits, label)
    print(f"DICE SCORE: {ds.item()} - DICE LOSS: {dl.item()}")
    print(f"CLASS-WISE DICE SCORE: \n {class_ds}")
    print(f"CLASS-WISE DICE LOSS: \n {class_dl}")
    
    iou = IOU(num_classes=8)
    iou_score, class_iou = iou(logits, label)
    print(f"IOU SCORE: {iou_score.item()} - IOU LOSS: {1 - iou_score.item()}")
    print(f"CLASS-WISE IOU SCORE: \n {class_iou}")
    
rot_val()
