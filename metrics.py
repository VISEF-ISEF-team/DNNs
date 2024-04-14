import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
import matplotlib.pyplot as plt

__all__ = ['Dice loss', 'Cross entropy', 'Focal loss', 'Dice cross entropy', 'Binary dice loss']


class IOU(nn.Module):
    '''
    Calculate Intersection over Union (IoU) for semantic segmentation.
    
    Args:
        logits (torch.Tensor): Predicted tensor of shape (batch_size, num_classes, height, width)
        target (torch.Tensor): Ground truth tensor of shape (batch_size, height, width)
        num_classes (int): Number of classes

    Returns:
        tensor: Mean Intersection over Union (IoU) for the batch.
        list: List of IOU score for each class
    '''
    def __init__(self, num_classes):
        super(IOU, self).__init__()
        self.num_classes = num_classes
        
    def forward(self, logits, target):
        target = target.unsqueeze(0)
        ious = []
        for cls in range(1, self.num_classes):
            pred_mask = (logits.argmax(dim=1) == cls)
            target_mask = (target == cls)
                            
            intersection = (pred_mask & target_mask).sum().float()
            union = (pred_mask | target_mask).sum().float()
            
            if union == 0: iou = 1.0 
            else: iou = (intersection / union).item()
            ious.append(iou)
        
        mean_iou = sum(ious) / (self.num_classes - 1)
        return torch.tensor(mean_iou), ious

class Dice(nn.Module):
    '''
    Calculate Dice score and Dice loss for semantic segmentation.
    
    Args:
        output (torch.Tensor): Predicted tensor of shape (batch_size, num_classes, height, width)
        target (torch.Tensor): Ground truth tensor of shape (batch_size, height, width)
        num_classes (int): Number of classes 
        
    Returns:
        tensor: Mean dice score over classes
        tensor: Mean dice loss over classes
        list: dice score for each classes
        listL dice loss for each classes
    '''
    
    def __init__(self, num_classes):
        super(Dice, self).__init__()
        self.num_classes = num_classes
        
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.num_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - dice
        return dice, loss

    def forward(self, inputs, target, weight=None):
        target = self._one_hot_encoder(target)
        
        if weight is None:
            weight = [1] * self.num_classes
        
        assert inputs.size() == target.size(), 'Predict {} & Target {} shape do not match'.format(inputs.size(), target.size())
        
        class_wise_dice_score = []
        class_wise_dice_loss = []
        LOSS = 0.0
        DICE = 0.0
        
        for i in range(1, self.num_classes):  # Start from 1 to exclude background class (class 0)
            dice, loss = self._dice_loss(inputs[:, i], target[:, i])            
            class_wise_dice_score.append(dice.item())
            class_wise_dice_loss.append(loss.item())
            DICE += dice
            LOSS += loss
            
        num_valid_classes = self.num_classes - 1  # Exclude background class
        
        return DICE / num_valid_classes, LOSS / num_valid_classes, class_wise_dice_score, class_wise_dice_loss
    
    
class BinaryDiceLoss(nn.Module):
    '''
    Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    '''
    def __init__(self, smooth=1, p=2):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        # predict = predict.contiguous().view(predict.shape[0], -1)
        # target = target.contiguous().view(target.shape[0], -1)

        # num = torch.sum(torch.mul(predict, target))*2 + self.smooth
        # den = torch.sum(predict.pow(self.p) + target.pow(self.p)) + self.smooth

        # dice = num / den
        # loss = 1 - dice
        # return loss
        
        smooth = 1e-5
        intersect = torch.sum(predict * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(predict * predict)
        dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - dice
        return loss
        

class DiceLoss(nn.Module):
    '''
    Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    '''
    def __init__(self, weight=None, ignore_index=0, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        
        # print(predict.size(), target.size())
        
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        # predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i == self.ignore_index: continue
            dice_loss = dice(predict[:, i], target[:, i])
            print(f'DICE LOSS FOR CLASS {i}: {dice_loss}')
            if self.weight is not None:
                assert self.weight.shape[0] == target.shape[1], \
                    'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                dice_loss *= self.weights[i]
            total_loss += dice_loss

        return total_loss/ (target.shape[1]-1)    
    
    
def test2D():
    mask = torch.tensor(np.load('data/imagechd/p_labels/0001_0135.npy')).unsqueeze(0)
    mask2 = torch.tensor(np.load('data/imagechd/p_labels/0001_0136.npy')).unsqueeze(0)
    num_classes = 8
    y_logits = torch.zeros((num_classes,) + mask2.shape[1:], dtype=torch.float)
    y_logits.scatter_(0, mask2.long(), 1.0)
    y_logits = y_logits.unsqueeze(0)

    print(y_logits.size(), mask.size())

    '''
    y_logits: (1,8,512,512) -> (batch, classes, width, height)
    label: (1,512,512) -> (batch, width, height)
    '''

    dice = Dice(num_classes=num_classes)
    ds, ls, wise_ds, wise_ls = dice(y_logits, mask)
    print(ds, ls, wise_ds, wise_ls)

    iou = IOU(num_classes)
    res, iou_list = iou(y_logits, mask)
    print("Mean IoU:", res)


    ce = nn.CrossEntropyLoss()
    ce_loss = ce(y_logits, mask[:].long())
    print(ce_loss)

    print(ce_loss.item()*0.2 + (1-res.item())*0.4 + ls.item()*0.4)
    
    
def test3D():
    label = sitk.GetArrayFromImage(sitk.ReadImage('data/imagechd/256_labels/0001.nii.gz')).astype(np.int64)
    
    # Real encoded label from train loader
    unique_values = np.unique(label)
    encoded_label = np.zeros((len(unique_values),) + label.shape, dtype=np.int64)
    for i, value in enumerate(unique_values): encoded_label[i][label == value] = 1
    
    for clx in range(8):
        print(encoded_label[clx, 100, 100, 100])
    print()
    encoded_label = torch.tensor(encoded_label).unsqueeze(0)
        
    # Real logits from model
    logits = np.zeros((len(unique_values),) + label.shape, dtype=np.float32)
    for i, value in enumerate(unique_values): 
        logits[i][label == value] = 123
        logits[i][label != value] = 1.7
        
    for clx in range(8):
        print(logits[clx, 100, 100, 100])
    print()
    logits = torch.tensor(logits).unsqueeze(0)    
    _, pred = torch.max(logits, dim=1)
    
    # encode the logits to make it like label
    pred = pred[0].detach().cpu().numpy()
    encoded_pred = np.zeros((len(unique_values),) + pred.shape)
    for i, value in enumerate(unique_values): encoded_pred[i][pred == value] = 1
    encoded_pred = torch.tensor(encoded_pred).unsqueeze(0)
        
    cost = DiceLoss()
    loss = cost(encoded_pred, encoded_label)
    print(loss)