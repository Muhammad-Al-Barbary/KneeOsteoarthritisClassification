import os 
from configs import config
from monai.data import Dataset, DataLoader
from data.transforms import train_transforms, test_transforms
import torch
import random
import numpy as np
import monai 

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    monai.utils.set_determinism(seed=seed)
    
def get_data(data_path, mode):
    data_path = os.path.join(data_path,mode)
    dataset = []
    for class_folder in os.listdir(data_path):
        for image_path in os.listdir(os.path.join(data_path,class_folder)):
            dataset.append({'image_path': os.path.join(data_path,class_folder,image_path), 'image': os.path.join(data_path,class_folder,image_path), 'label': int(class_folder)})
    return dataset

def get_loader(data_path, transforms = None, mode = 'training', shuffle = False, batch_size = 1, seed=42):
    set_seed(seed)
    data = get_data(data_path, mode)
    dataset = Dataset(data, transforms)
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
    return dataloader

dataset_path = config.data['path']
train_loader = get_loader(dataset_path, transforms=train_transforms, mode='train', shuffle=config.data['shuffle'], batch_size=config.data['batch_size'], seed=config.seed)
val_loader = get_loader(dataset_path, transforms=test_transforms, mode='val', shuffle=False, batch_size=config.data['batch_size'])
test_loader = get_loader(dataset_path, transforms=test_transforms, mode='test', shuffle=False, batch_size=config.data['batch_size'])