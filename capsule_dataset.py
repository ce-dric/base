import torch
import os
import os.path as osp
import numpy as np

import torchvision.transforms as T

from PIL import Image
from common import CLASSES, PROJ_DIR

class MVTEC_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, delimeter) -> None:
        super().__init__()
        self.data_path = osp.join(PROJ_DIR, 'dataset', dataset, delimeter)
        self.transform = T.ToTensor()
        self.images, self.labels = self.make_dataset()
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        label = self.labels[index]
        
        img = self.transform(img)
        
        return img, label
    
    def make_dataset(self):
        labels = os.listdir(self.data_path)
        
        image_list = []
        label_list = []
        for label in labels:
            for file in os.listdir(osp.join(self.data_path, label)):
                image_list.append(osp.join(self.data_path, label, file))
                label_list.append(list(CLASSES.keys())[list(CLASSES.values()).index(label)])
        
        return image_list, label_list

if __name__ == '__main__':
    dataset = MVTEC_Dataset('capsule', 'train')
    print(dataset.__len__())
