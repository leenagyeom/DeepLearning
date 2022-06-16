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
train_data = dataset.CustomDataset(path = configer.data_path, mode ="train", transform=data_transform['train'])
test_data = dataset.CustomDataset(path = configer.data_path, mode ="val", transform=data_transform['test'])


### 3. dataloader setting
train_loader = DataLoader(train_data, batch_size=configer.batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=configer.batch_size, shuffle=True, drop_last=True)
# for data, target in test_loader:
#     print(data, target)
# exit()


### 4. model call
net, image_size = models_build.initialize_model("resnet", num_classes=configer.nc)
# print(net, image_size)
# exit()


### 5. hyper parameter call - loss function call, optimizer, learning schedule
criterion = configer.criterion
optimizer = optim.SGD(net.parameters(), lr=configer.lr, momentum=0.9)
lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma = 0.1)


# 과제 : train loop 구성해서 학습돌리고 학습 결과를 프린트
### 6. train loop function call


### 7. test loop function call
