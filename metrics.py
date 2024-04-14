import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
import matplotlib.pyplot as plt

__all__ = ['Dice loss', 'Cross entropy', 'Focal loss', 'Dice Iou Cross entropy', 'Binary dice loss']


class IOU(nn.Module):
    '''
    Calculate Intersection over Union (IoU) for semantic segmentation.
    
    Args:
        logits (torch.Tensor): Predicted tensor of shape (batch_size, num_classes, height, width, (depth))
        target (torch.Tensor): Ground truth tensor of shape (batch_size, height, width, (depth))
        num_classes (int): Number of classes

    Returns:
        tensor: Mean Intersection over Union (IoU) for the batch.
        list: List of IOU score for each class
    '''
    def __init__(self, num_classes, ignore_index=[0]):
        super(IOU, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
    def forward(self, logits, target):
        pred = logits.argmax(dim=1)         # 1,256,256,256
        target = target.argmax(dim=1)        # 8,256,256,256
        ious = []
        for cls in range(self.num_classes):
            if cls in self.ignore_index: continue
            pred_mask = (pred == cls)
            target_mask = (target == cls)
                            
            intersection = (pred_mask & target_mask).sum().float()
            union = (pred_mask | target_mask).sum().float()
            
            if union == 0: iou = 1.0 
            else: iou = (intersection / union).item()
            ious.append(iou)
        
        mean_iou = sum(ious) / (self.num_classes - len(self.ignore_index))
        return torch.tensor(mean_iou), ious

    
class BinaryDice(nn.Module):
    '''
    Calculate Binary Dice score and Dice loss for binary segmentation or each class in Multiclass segmentation
    
    Args:
        logits (torch.Tensor): Predicted tensor of shape (batch_size, height, width, (depth))
        target (torch.Tensor): Ground truth tensor of shape (batch_size, height, width. (depth))
        
    Returns:
        tensor: Dice score
        tensor: Dice loss
    '''
    def __init__(self, smooth=1e-5, p=2):
        super(BinaryDice, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, logits, target):
        assert logits.shape[0] == target.shape[0], "logits & Target batch size don't match"
        smooth = 1e-5
        intersect = torch.sum(logits * target)        
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(logits * logits)
        dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - dice
        return dice, loss
        

class Dice(nn.Module):
    '''
    Calculate Dice score and Dice loss for multiclass semantic segmentation
    
    Args:
        output (torch.Tensor): Predicted tensor of shape (batch_size, num_classes, height, width, (depth))
        target (torch.Tensor): Ground truth tensor of shape (batch_size, height, width, (depth))
        num_classes (int): Number of classes 
        
    Returns:
        tensor: Mean dice score over classes
        tensor: Mean dice loss over classes
        list: dice score for each classes
        listL dice loss for each classes
    '''
    def __init__(self, num_classes, weight=None, ignore_index=[0]):
        super(Dice, self).__init__()
        self.num_classes = num_classes
        self.weight = weight
        self.ignore_index = ignore_index
        self.binary_dice = BinaryDice()

    def forward(self, logits, target):
        assert logits.shape == target.shape, 'logits & Target shape do not match'
        logits = F.softmax(logits, dim=1)
        
        DICE, LOSS = 0.0, 0.0
        CLS_DICE, CLS_LOSS = [], []
        for clx in range(target.shape[1]):
            if clx in self.ignore_index: continue
            dice, loss = self.binary_dice(logits[:, clx], target[:, clx])
            CLS_DICE.append(dice.item())
            CLS_LOSS.append(loss.item())
            if self.weight is not None: dice *= self.weights[clx]
            DICE += dice
            LOSS += loss

        num_valid_classes = self.num_classes - len(self.ignore_index)
        return DICE / num_valid_classes, LOSS / num_valid_classes, CLS_DICE, CLS_LOSS
    
    
def test2D():
    '''
    logits: (1,8,512,512) -> (batch, classes, width, height)
    label:  (1,8,512,512) -> (batch, classes, width, height)  -> encoded
    '''
    
    num_classes = 8
    
    # Real encoded label from train loader 
    mask = np.load('data/imagechd/p_r_labels/0001_0135.npy')
    encoded_label = np.zeros( (num_classes, ) + mask.shape)
    for i in range(num_classes): 
        encoded_label[i][mask == i] = 1
        
    for clx in range(num_classes):
        print(encoded_label[clx, 100, 100])
    print()
    encoded_label = torch.tensor(encoded_label).unsqueeze(0)
    
    # (prepare) Real predictions from model
    mask = np.load('data/imagechd/p_r_labels/0001_0135.npy').astype(np.int64)
    logits = np.zeros( (num_classes, ) + mask.shape, dtype=np.float32)
    for i in range(num_classes): 
        logits[i][mask == i] = 6
        logits[i][mask != i] = 0.12

    for clx in range(num_classes):
        print(logits[clx, 100, 100])
    print()
    logits = torch.tensor(logits).unsqueeze(0)

    dice_func = Dice(num_classes)
    ds, dsl, cls_ds, cls_dsl = dice_func(logits, encoded_label)
    print(f'DICE SCORE: {ds} - DICE LOSS: {dsl}')

    iou_func = IOU(num_classes)
    iou, cls_iou = iou_func(logits, encoded_label)
    print("IOU:", iou)

    ce = nn.CrossEntropyLoss()
    ce_loss = ce(logits, encoded_label)
    print(f'CE LOSS: {ce_loss}')
    
    
def test3D():
    '''
    logits: (1,8,512,512,512) -> (batch, classes, width, height, depth)
    label:  (1,8,512,512,512) -> (batch, classes, width, height, depth)  -> encoded
    '''
    label = sitk.GetArrayFromImage(sitk.ReadImage('data/imagechd/256_labels/0001.nii.gz')).astype(np.int64)
    num_classes = 8
    
    # Real encoded label from train loader
    encoded_label = np.zeros((num_classes,) + label.shape, dtype=np.float32)
    for i in range(num_classes):
        encoded_label[i][label == i] = 1
    
    for clx in range(num_classes):
        print(encoded_label[clx, 100, 100, 100])
    print()
    encoded_label = torch.tensor(encoded_label).unsqueeze(0)
        
    # (prepare) Real logits from model
    logits = np.zeros((num_classes,) + label.shape, dtype=np.float32)
    for i in range(num_classes):
        logits[i][label == i] = 6
        logits[i][label != i] = 0.2
        
    for clx in range(8):
        print(logits[clx, 100, 100, 100])
    print()
    logits = torch.tensor(logits).unsqueeze(0)
    
    dice_func = Dice(num_classes)
    ds, dsl, cls_ds, cls_dsl = dice_func(logits, encoded_label)
    print(f'DICE SCORE: {ds} - DICE LOSS: {dsl}')

    iou_func = IOU(num_classes)
    iou, cls_iou = iou_func(logits, encoded_label)
    print("IOU:", iou)

    ce = nn.CrossEntropyLoss()
    ce_loss = ce(logits, encoded_label)
    print(f'CE LOSS: {ce_loss}')