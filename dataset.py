import os
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, image_paths, label_paths, image_ext='.npy', label_ext='.npy'):
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
            ├── images
            |   ├── 0001_0001.npy
            │   ├── 0001_0002.npy
            │   ├── 0001_0003.npy
            │   ├── ...
            |
            └── labels
                ├── 0001_0001.npy
                ├── 0001_0002.npy
                ├── 0001_0003.npy
                ├── ...     
        '''
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.image_ext = image_ext
        self.label_ext = label_ext
        self.length = len(image_paths)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        image = np.load(self.image_paths[index]) / 255.0
        label = np.load(self.label_paths[index])
        
        return image, label