import os
from torch.utils.data import Dataset
import numpy as np
import torch
from scipy.ndimage import zoom
import SimpleITK as sitk

class CustomDataset(Dataset):
    def __init__(self, image_paths, label_paths, img_size, image_ext='.npy', label_ext='.npy'):
        '''
        Args:
            image_paths: Image file paths.
            label_paths: Label file paths.
            image_ext (str): Image file extension.
            label_ext (str): Label file extension.
            num_classes (int): Number of classes.
        
        Note:
            Make sure to process the data into this structures
            <dataset name>
            ├── p_images
            |   ├── 0001_0001.npy
            │   ├── 0001_0002.npy
            │   ├── 0001_0003.npy
            │   ├── ...
            |
            └── p_labels
                ├── 0001_0001.npy
                ├── 0001_0002.npy
                ├── 0001_0003.npy
                ├── ...     
        '''
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.img_size = img_size
        self.image_ext = image_ext
        self.label_ext = label_ext
        self.length = len(image_paths)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        image = np.load(self.image_paths[index])
        label = np.load(self.label_paths[index])
        
        x, y = image.shape
        if x != self.img_size and y != self.img_size:
            image = zoom(image, (self.img_size / x, self.img_size / y), order=0)
            label = zoom(label, (self.img_size / x, self.img_size / y), order=0)
        
        return image, label
    
    
class CustomDataset3D(Dataset):
    def __init__(self, image_paths, label_paths, ext='.nii.gz'):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.ext = ext
        self.length = len(image_paths)
        
    def __len__(self):
        return self.length
    
    def encode(self, label):
        unique_values, _ = np.unique(label, return_counts=True)
        vol = []
        for value in unique_values: 
            label_array = np.copy(label) 
            label_array[np.where(label_array != value)] = 0 
            vol.append(label_array)
        
        vol = np.stack(vol, dim=0)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_path = self.label_paths[index]
        
        image = sitk.GetArrayFromImage(sitk.ReadImage(image_path, sitk.sitkFloat32))
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        
        image /= np.max(image)
        
        image = np.transpose(image, (2, 1, 0))
        label = np.transpose(label, (2, 1, 0))
        
        image = image[np.newaxis, :, :, :]
        label = self.encode(label)
        
        return torch.from_numpy(image), torch.from_numpy(label)