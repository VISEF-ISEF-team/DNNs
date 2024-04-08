import torch
import torch.nn as nn
from collections import OrderedDict
from tqdm import tqdm
from metrics import IOU
import csv

def write_csv(path, data):
    with open(path, mode='a', newline='') as file:
        iteration = csv.writer(file)
        iteration.writerow(data)
    file.close()

def trainer(config, train_loader, optimizer, model, ce, dice, iou):
    # one epoch        
    steps = len(train_loader)
    pbar = tqdm(total=steps)
    
    fieldnames = ['CE Loss', 'Dice Score', 'Dice Loss', 'IoU Score', 'IoU Loss', 'Total Loss']
    write_csv(f'outputs/{config.name}/iteration.csv', fieldnames)
    
    total_ce_loss, total_dice_score, total_dice_loss, \
    total_iou_score, total_iou_loss, total_loss = 0,0,0,0,0,0
    
    for iteration, (input, target) in enumerate(train_loader):
        input = input.unsqueeze(1).cuda()
        target = target.cuda()
            
        if config.type == 'Transformer':
            logits = model(input)
        
            ce_loss = ce(logits, target[:].long())
            dice_loss, dice_score, class_dice_score, class_dice_loss = dice(logits, target)
            iou_score, class_iou =  iou(logits, target)
            loss = 0.2 * ce_loss + 0.3 * dice_loss + 0.5 * (1 - iou_score)
            
            total_ce_loss += ce_loss.item()
            total_dice_score += dice_score.item()
            total_dice_loss += dice_loss.item()
            total_iou_score += iou_score.item()
            total_iou_loss += 1.0-iou_score.item()
            total_loss += loss.item()
                
            write_csv(f'outputs/{config.name}/iteration.csv', [
                ce_loss.item(),
                dice_score.item(),
                dice_loss.item(),
                iou_score.item(),
                1.0-iou_score.item(),
                loss.item()
            ])
            
            write_csv(f'outputs/{config.name}/ds_class_iter.csv', class_dice_score)
            write_csv(f'outputs/{config.name}/dl_loss_iter.csv', class_dice_loss)
            write_csv(f'outputs/{config.name}/iou_class_iter.csv', class_iou)
                                            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    return OrderedDict([
        ('ce_loss', total_ce_loss / steps),
        ('dice_score', total_dice_score / steps),
        ('dice_loss', total_dice_loss / steps),
        ('iou_score', total_iou_score / steps),
        ('iou_loss', total_dice_loss / steps),
        ('loss', total_loss / steps)
    ])