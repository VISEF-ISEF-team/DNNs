import os
from os.path import join as pj
import SimpleITK as sitk
import numpy as np
import torch
from skimage.transform import resize
import matplotlib.pyplot as plt
from glob import glob

def save(ds_dir, folder, data_path):
    folder_path = pj(ds_dir, 'p_' + folder)
    if not os.path.exists(folder_path): os.makedirs(folder_path)
    if folder == 'labels': vol = sitk.GetArrayFromImage(sitk.ReadImage(data_path))
    else: vol = sitk.GetArrayFromImage(sitk.ReadImage(data_path, sitk.sitkFloat32))
    for index in range(vol.shape[0]):
        path_to_save = pj(ds_dir, 'p_' + folder, f'{data_path[-11:-7]}_{index+1:04d}.npy')
        if folder == 'images':
            norm_vol = vol[index] / np.max(vol[index])
            if not os.path.exists(path_to_save): np.save(path_to_save, norm_vol)
        else: 
            if not os.path.exists(path_to_save): np.save(path_to_save, vol[index])
        

def run(ds_dir, ext='.nii.gz', start=None, end=None):
    image_ids = glob(pj(ds_dir, 'images', f'*{ext}'))
    label_ids = glob(pj(ds_dir, 'labels', f'*{ext}'))
    if start != None and end != None: 
        image_ids = image_ids[start:end]
        label_ids = label_ids[start:end]
    print(f'Length detected: {len(image_ids)}')
    
    for data_path in image_ids:
        save(ds_dir, 'images', data_path)
    for data_path in label_ids:
        save(ds_dir, 'labels', data_path)

    
run(
    ds_dir='../data/imagechd',
    start=0, end=10
)