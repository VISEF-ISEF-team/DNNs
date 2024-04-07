import torch
import torch.nn as nn
from collections import OrderedDict
from tqdm import tqdm
from metrics import IOU

def trainer(config, train_loader, optimizer, model, ce, dice):
        
    pbar = tqdm(total=len(train_loader))
    for index, (input, target) in enumerate(train_loader):
        input = input.unsqueeze(1).cuda()
        target = target.cuda()
            
        if config.type == 'Transformer':
            output = model(input)
        
            ce_loss = ce(output, target[:].long())
            dice_loss, dice_score, dice_class_wise = dice(output, target, softmax=True)
            dice_ce_loss = 0.5 * ce_loss + 0.5 * dice_loss 
            
            iou_score = IOU(output, target)
            print(iou_score)
            return
            
            print(ce_loss, dice_loss, dice_ce_loss, dice_score, iou_score)
            print(dice_class_wise)
            return
            
            optimizer.zero_grad()
            dice_ce_loss.backward()
            optimizer.step()
        
        return

    # return OrderedDict([
    #     ('train_ce_loss', ce_loss),
    #     ('train_dice_loss', dice_loss),
    #     ('train_dice_score', dice_score),
    #     ('train_dice_class_wise', dice_class_wise)
    #     ('train_iou', iou_score),
    #     ('train_dice_ce_loss', dice_ce_loss)
    # ])