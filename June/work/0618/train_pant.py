import torch.cuda
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import utils_file, dataset, configer, models_build

"""
models_build.py 학습할 모델 build 파일
utils_file.py   여러 잡동사니 (Image Show 필요한 함수 구현)
dataset.py      학습 데이터를 가져오기 위한 데이터셋 구성 (CIFAR, json 등)
configer.py     Hyper Parameter 값 셋팅 (epoch, lr, momentum, gamma 등)
"""

device = configer.device


### 1. augmentation setting
data_transform = utils_file.data_augmentation()


### 2. dataset setting
# 데이터 경로, mode, transform 인자 필요
datasets = {x : dataset.CustomDataset(path = configer.data_path, mode = x, transform=data_transform[x]) for x in ['train', 'test']}


### 3. dataloader setting
dataloders = {x : DataLoader(datasets[x], batch_size=configer.batch_size, shuffle=True, drop_last=True) for x in ['train', 'test']}
dataset_sizes = {x : len(datasets[x]) for x in ['train', 'test']}


### 4. model call
modelname = "resnet34"
net, image_size = models_build.initialize_model(modelname, num_classes=configer.nc)
net = net.to(device)


### 5. hyper parameter call - loss function call, optimizer, learning schedule
criterion = configer.criterion
optimizer = optim.SGD(net.parameters(), lr=configer.lr, momentum=0.9)
lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma = 0.1)


### 6. train loop function call
model_ft, p_data = utils_file.train_model(net, criterion, optimizer,  dataloders, dataset_sizes, lr_schedule, configer.num_epochs)
utils_file.save_model(model_ft, modelname)
utils_file.loss_acc_visualize(p_data, modelname)
utils_file.visual_predict(model_ft, modelname, datasets['test'])

### 7. test loop function call