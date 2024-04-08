import os
from os.path import join as pj
import SimpleITK as sitk
import numpy as np
import torch
from skimage.transform import resize
import matplotlib.pyplot as plt
from glob import glob


def run(ds_dir):
    dir = glob(pj(ds_dir, 'train_npz', '*.npz'))
    for path in dir:
        data = np.load(path)
        image, label = data['image'], data['label']
        
        if not os.path.exists(pj(ds_dir, 'p_images')): os.makedirs(pj(ds_dir, 'p_images'))
        if not os.path.exists(pj(ds_dir, 'p_labels')): os.makedirs(pj(ds_dir, 'p_labels'))
        
        case = path[-17:-13]
        slice = path[-7:-4]
        path_image = pj(ds_dir, 'p_images', f'{case}_{int(slice):04d}.npy')
        path_label = pj(ds_dir, 'p_labels', f'{case}_{int(slice):04d}.npy')
        if not os.path.exists(path_image): np.save(path_image, image)
        if not os.path.exists(path_label): np.save(path_label, label)

run('../data/Synapse')

    
    
