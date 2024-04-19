import argparse
import yaml
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import zoom
from metrics import Dice, IOU
from skimage.transform import resize as skires
import SimpleITK as sitk
import torch.nn.functional as F

# Training graphs
def visualize(epochs, scores, legends, x_label, y_label, title, config):
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan'] 
    for score, legend, color in zip(scores, legends, colors):
        plt.plot(epochs, score, color, label=legend)
        
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(f"outputs/{config['name']}/graph1.jpeg")
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
    visualize(iters, [metrics[4], metrics[5], metrics[6], metrics[7]],  [fields[4], fields[5], fields[6], fields[7]],
            'Epochs', 'Scores', 'Training results', config)
    

# Prediction comparision
def compare(gt, label):
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))  
    axes[0].set_title('Label')
    axes[0].imshow(gt) 
    axes[0].axis('off')
    
    axes[1].set_title('Prediction')
    axes[1].imshow(label) 
    axes[1].axis('off') 
    
    plt.tight_layout() 
    plt.show()
    
def rot_val():
    model = torch.load('outputs\imagechd_NestedTransUnetRot_bs12_ps16_epo150_hw128\model.pth')
    vol = []
    labels = []
    for index in range(120, 132, 1):
        input = np.load(f'data/imagechd/p_images/0004_0{index}.npy')
        label = np.load(f'data/imagechd/p_labels/0004_0{index}.npy')
        
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
        compare(label, pred)
    
def val_each(vol_index, index):
    # Input
    model = torch.load('outputs/imagechd_TransUNet_bs12_ps16_R50-ViT-B_16_epo150_hw128/model.pth')
    input = np.load(f'data/imagechd/p_r_images/{vol_index:04d}_{index:04d}.npy')
    label = np.load(f'data/imagechd/p_r_labels/{vol_index:04d}_{index:04d}.npy')
    
    # Dataloader
    x,y = input.shape
    img_size = 128
    input = zoom(input, (img_size / x, img_size / y), order=0)
    label = zoom(label, (img_size / x, img_size / y), order=0)
    
    # Prediction
    input = torch.tensor(input).unsqueeze(0).unsqueeze(0).cuda()
    logits = model(input)
    val, pred = torch.max(logits, dim=1)
    
    # detach
    pred = pred[0].detach().cpu().numpy()
    return pred, label

def save_vol(vol, path):
    vol = np.transpose(vol, (2, 1, 0))
    affine = np.eye(4)
    nifti_file = nib.Nifti1Image(vol.astype(np.int8), affine)
    nib.save(nifti_file, path)
    
def resize_vol(vol, new_size):
    return skires(vol, new_size, order=1, preserve_range=True, anti_aliasing=False)
    
def val_pipeline():
    vol = []
    label = []
    vol_index = 22
    model = 'TransUNet'
    for index in range(1, 221+1, 1):
        slice, mask = val_each(vol_index, index)
        vol.append(slice)
        label.append(mask)
        
    vol = resize_vol(np.array(vol), (128, 128, 128))
    label = resize_vol(np.array(label), (128, 128, 128))
    save_vol(np.array(vol), f'reconstruction/{vol_index:04d}_pred_{model}_2.nii.gz')
    save_vol(np.array(label), f'reconstruction/{vol_index:04d}_label_{model}_2.nii.gz')
    
    
def prediction(id, path):
    image = sitk.GetArrayFromImage(sitk.ReadImage(path, sitk.sitkFloat32))
    model_axial = torch.load('outputs/imagechd_NestedTransUnetRot_bs12_ps16_epo150_hw128/model.pth')
    # model_sagittal = torch.load('outputs/imagechd_TransUNet_bs24_ps16_R50-ViT-B_16_epo200_hw128_sagittal/model.pth')
    # model_coronal = torch.load('outputs/imagechd_TransUNet_bs24_ps16_R50-ViT-B_16_epo200_hw128_coronal/model.pth')
    
    axial_vol = []
    sagittal_vol = []
    coronal_vol = []
    
    for index in range(image.shape[0]):
        # axial
        slice = image[index,:,:]
        slice = slice / np.max(slice)
        slice = torch.tensor(slice).unsqueeze(0).unsqueeze(0).cuda()
        output = model_axial(slice)
        _, pred = torch.max(output, dim=1)        
        pred = pred.squeeze(0).detach().cpu().numpy()
        axial_vol.append(pred)
        
        # # sagittal
        # slice = image[:,index,:]
        # slice = slice / np.max(slice)
        # slice = torch.tensor(slice).unsqueeze(0).unsqueeze(0).cuda()
        # output = model_sagittal(slice)
        # _, pred = torch.max(output, dim=1)        
        # pred = pred.squeeze(0).detach().cpu().numpy()
        # sagittal_vol.append(pred)
        
        # # sagittal
        # slice = image[:,:,index]
        # slice = slice / np.max(slice)
        # slice = torch.tensor(slice).unsqueeze(0).unsqueeze(0).cuda()
        # output = model_coronal(slice)
        # _, pred = torch.max(output, dim=1)        
        # pred = pred.squeeze(0).detach().cpu().numpy()
        # coronal_vol.append(pred)
        
    
    save_vol(np.array(axial_vol), 'reconstruction/0001_NestedTransUnetRot_new.nii.gz')
    
def val_rot(path):
    model = torch.load('outputs/imagechd_NestedTransUnetRot_bs12_ps16_epo150_hw128/model.pth')
    image = sitk.GetArrayFromImage(sitk.ReadImage(path, sitk.sitkFloat32))
    image_tensor = torch.from_numpy(image).float().cuda()
    segment_size = 12
    results = []

    with torch.no_grad():
        for i in range(0, image_tensor.shape[0], segment_size):
            # Extract segment
            input = image_tensor[i:i+segment_size, :, :].unsqueeze(1).cuda()
            
            # Predict using your model
            output = model(input)
            _, pred = torch.max(output, dim=1)
            
            results.append(pred)

    # Concatenate results along the first dimension
    final_output = torch.cat(results, dim=0)
    final_output = final_output.detach().cpu().numpy()
    save_vol(final_output, 'reconstruction/0001_NestedTransUnetRot_new.nii.gz')
    

id = 1
path = f'data/imagechd/128_images_axial/{id:04d}.nii.gz'
# prediction(id, path)
val_rot(path)
