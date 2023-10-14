import os
from pathlib import Path

import torch
import torchvision.transforms as TF
from torch.utils.data import DataLoader, Dataset

import numpy as np
from skimage import io

from utils.conversions import get_dataframe_annotations
from utils.transforms import RandomCrop, Rescale, ToTensor

def get_specified_transforms_custom():
    data_transforms = {
        'train': TF.Compose([
            RandomCrop(224),
            ToTensor(),
        ]),
        'val': TF.Compose([
            Rescale(256),
            ToTensor(),
        ]),
        'test': TF.Compose([
            Rescale(256),
            ToTensor(),
        ]),
    }
    return data_transforms

def get_specified_transforms():
    transform = TF.Compose([TF.ToTensor(),
     TF.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    
    data_transforms = {
        'train': transform,
        'val': transform,
        'test': transform,
    }
    
    return data_transforms


def load_dataset(annotation, root: Path):
    # Convert to more transparent format
    root = root / annotation['data_path']
    datasets_paths = dict({"train": root / annotation['train_c'],
                           "val": root / annotation['val_c'],
                           "test": root / annotation['test_c']})
    images_path = annotation['images_path']
    labels_path = annotation['labels_path']
    transforms = get_specified_transforms()
    
    annotations_dfs = {dataset_name: get_dataframe_annotations(datasets_paths[dataset_name] / labels_path,
                                                 datasets_paths[dataset_name] / images_path) 
                        for dataset_name in datasets_paths.keys()}
    datasets = {dataset_name: CustomDataset(annotations_dfs[dataset_name], 
                                            datasets_paths[dataset_name] / images_path, 
                                            transform=transforms[dataset_name]) 
                for dataset_name in datasets_paths.keys()}

    return datasets

def get_data_loaders(datasets, datasets_names, batch_size, shuffle=True, num_workers=1):
    dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size,
                                             shuffle=shuffle, num_workers=num_workers)
              for x in datasets_names}

    return dataloaders


class CustomDataset(Dataset):
    """Custom dataset defined with PyTorch Dataset as base class"""

    def __init__(self, annotations_frame, img_dir, transform=None):
        """
        Arguments:
            annotations_frame (DataFrame): DataFrame with annotations
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotations_frame = annotations_frame
        self.root_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations_frame)

    # Function for reading images
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.annotations_frame.iloc[idx, 0])
        img = io.imread(img_name) 
        labels = int(self.annotations_frame.iloc[idx, 3])
        boxes = np.array([self.annotations_frame.iloc[idx, 4:]], dtype=float).reshape(-1)

        if self.transform:
            img = self.transform(img)
            labels = torch.from_numpy(np.array(labels)).type(torch.LongTensor)
            boxes = torch.from_numpy(boxes).float()
            
        return img, labels, boxes#sample['img'], sample['labels']