import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import zoom
from viz import compare
from metrics import Dice, IOU

model = torch.load('outputs/Synapse_TransUnet_bs10_ps16_R50-ViT-B_16_epo10_hw256/model.pth')

input = np.load('data/Synapse/p_images/0005_0060.npy')
label = np.load('data/Synapse/p_labels/0005_0060.npy')

x,y = input.shape
img_size = 256
input = zoom(input, (img_size / x, img_size / y), order=0)
label = zoom(label, (img_size / x, img_size / y), order=0)

input = torch.tensor(input).unsqueeze(0).unsqueeze(0).cuda()

print(input.size())
output = model(input)

val, pred = torch.max(output, dim=1)
print(pred.size())
compare(
    label,
    pred[0].detach().cpu().numpy(),     
)

label = torch.tensor(label).unsqueeze(0).cuda()

print(torch.unique(label))

dl = Dice(num_classes=9)
ds, dl, class_ds, class_dl = dl(output, label)
print(ds.item(), dl.item())
print(class_ds)
print(class_dl)
