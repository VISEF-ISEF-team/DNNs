import torch
import torch.nn as nn

__all__ = ['Dice loss', 'Cross entropy', 'Focal loss', 'Dice cross entropy']


class Dice(nn.Module):
    def __init__(self, n_classes):
        super(Dice, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
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

    def forward(self, inputs, target, weight=None, softmax=True):
        
        if softmax: inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None: weight = [1] * self.n_classes
        
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice_score = []
        loss = 0.0
        dice = 0.0
        for i in range(0, self.n_classes):
            dice, loss = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice_score.append(dice.item())
            loss += loss * weight[i]
            dice += dice
            
        average_loss = loss / self.n_classes
        average_đice = dice / self.n_classes
        
        return average_loss, average_đice, class_wise_dice_score
    
    
def IOU(output, target, softmax=True):
    smooth = 1e-5

    values, pred = torch.max(output, dim=1)
    pred = pred.data.cpu().numpy()
    target = target.data.cpu().numpy()

    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def IOU(output: torch.Tensor, label: torch.Tensor):
    SMOOTH = 1e-6
    values, pred = torch.max(output, dim=1)
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    output = output.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (output & label).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (output | label).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch

