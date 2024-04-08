import numpy as np
import torch
import torch.nn as nn

__all__ = ['Dice loss', 'Cross entropy', 'Focal loss', 'Dice cross entropy']


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
    
    
    
def test():
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