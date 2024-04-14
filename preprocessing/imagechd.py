import os
from os.path import join as pj
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from glob import glob
from scipy.ndimage import zoom
from skimage.transform import resize as skires

# 1. Save .npy files from .nii.gz data
def save(ds_dir, folder, data_path):
    folder_path = pj(ds_dir, 'p_' + folder)
    if not os.path.exists(folder_path): os.makedirs(folder_path)
    if folder == 'r_labels': vol = sitk.GetArrayFromImage(sitk.ReadImage(data_path))
    else: vol = sitk.GetArrayFromImage(sitk.ReadImage(data_path, sitk.sitkFloat32))
    for index in range(vol.shape[0]):
        path_to_save = pj(ds_dir, 'p_' + folder, f'{data_path[-11:-7]}_{index+1:04d}.npy')
        if folder == 'r_images':
            norm_vol = vol[index] / np.max(vol[index])
            if not os.path.exists(path_to_save): np.save(path_to_save, norm_vol)
        else: 
            if not os.path.exists(path_to_save): np.save(path_to_save, vol[index])
        

def export_npy(ds_dir, ext='.nii.gz', start=None, end=None):
    image_ids = glob(pj(ds_dir, 'r_images', f'*{ext}'))
    label_ids = glob(pj(ds_dir, 'r_labels', f'*{ext}'))
    if start != None and end != None: 
        image_ids = image_ids[start:end]
        label_ids = label_ids[start:end]
    print(f'Length detected: {len(image_ids)}')
    
    for data_path in image_ids:
        save(ds_dir, 'r_images', data_path)
    for data_path in label_ids:
        save(ds_dir, 'r_labels', data_path)
        
        
# 2. Resize for 2D network (only predict on axial view)    
def save_vol(vol, type, path):
    vol = np.transpose(vol, (2, 1, 0))
    affine = np.eye(4)
    nifti_file = nib.Nifti1Image(vol.astype(np.int8), affine) if type == 'labels' else nib.Nifti1Image(vol, affine)
    nib.save(nifti_file, path)

def resize(vol, type, name):
    vol_new = []
    img_size = 256
    x, y = vol.shape[1], vol.shape[2]
    for index in range(vol.shape[0]):
        slice = vol[index]
        slice = zoom(slice, (img_size / x, img_size / y), order=0)
        vol_new.append(slice)
    
    save_vol(vol_new, type, f'../data/imagechd/r_{type}/{name}.nii.gz')

def process_size(ds_dir, ext='.nii.gz'):
    image_ids = glob(pj(ds_dir, 'images', f'*{ext}'))
    label_ids = glob(pj(ds_dir, 'labels', f'*{ext}'))
    
    if not os.path.exists(pj(ds_dir, 'r_images')): os.makedirs(pj(ds_dir, 'r_images'))
    if not os.path.exists(pj(ds_dir, 'r_labels')): os.makedirs(pj(ds_dir, 'r_labels'))

    for image_path in image_ids:
        image = sitk.GetArrayFromImage(sitk.ReadImage(image_path, sitk.sitkFloat32))
        resize(image, 'images', image_path[-11:-7])
    
    for label_path in label_ids:
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        resize(label, 'labels', label_path[-11:-7])
        
        
# 3. Resize for 2.5D and 3D network (no atifacts)
def resize_vol(vol, new_size):
    return skires(vol, new_size, order=1, preserve_range=True, anti_aliasing=False)

def seperate_classes(img):
    unique_values, _ = np.unique(img, return_counts=True)
    vol = []
    for value in unique_values: 
        label_array = np.copy(img) 
        label_array[np.where(label_array != value)] = 0 
        vol.append(label_array)
    return vol

def resize_label(label, new_size, name):
    vol_label = seperate_classes(label)
    for index in range(len(vol_label)):
        temp = vol_label[index]
        temp[np.where(temp != 0)] = 1
        temp = resize_vol(temp, new_size).astype(np.int8)
        vol_label[index] = temp
        
    vol_label = np.stack(vol_label[1:], axis=0)
    vol_label = np.argmax(vol_label, axis=0)
    save_vol(vol_label, 'labels', f'../data/imagechd/{new_size[0]}_labels/{name}.nii.gz')
    

def process_3D(ds_dir, ext='.nii.gz', new_size=(256,256,256)):
    image_ids = glob(pj(ds_dir, 'images', f'*{ext}'))
    label_ids = glob(pj(ds_dir, 'labels', f'*{ext}'))
    
    if not os.path.exists(pj(ds_dir, f'{new_size[0]}_images')): os.makedirs(pj(ds_dir, f'{new_size[0]}_images'))
    if not os.path.exists(pj(ds_dir, f'{new_size[0]}_labels')): os.makedirs(pj(ds_dir, f'{new_size[0]}_labels'))
    
    for label_path in label_ids:
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        resize_label(label, new_size, label_path[-11:-7])
    
    for image_path in image_ids:
        image = sitk.GetArrayFromImage(sitk.ReadImage(image_path, sitk.sitkFloat32))
        image = resize_vol(image, new_size)
        save_vol(image, 'images', f'{ds_dir}/{new_size[0]}_images/{image_path[-11:-7]}.nii.gz')
    
    

# process_3D(ds_dir='../data/imagechd')
    
export_npy(
    ds_dir='../data/imagechd',
    start=0, end=15
)