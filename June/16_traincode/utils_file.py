import torchvision.transforms as transforms
import torch
import os
import time
import copy
import configer
import dataset

"""
1. augmentation
2. train loop
3. valid loop
4. save model
5. eval test code
"""

def data_augmentation() : # 데이터 증가
    # data augmentation 함수
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    return data_transforms