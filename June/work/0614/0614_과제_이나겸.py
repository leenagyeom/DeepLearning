import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import seaborn as sns

from PIL import Image
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from timeit import default_timer as timer

classes = ['Downdog', 'Goddess', 'Plank', 'Tree', 'Warrior2']    # 라벨

# 1. 데이터 비율 확인 (mac, windows에 따라 split을 수정필요)
"""
os.walk로 path의 하위폴더를 탐색하다가 파일이 존재하고 root split 한 단어가 라벨에 있을 때,
파일갯수가 몇개인지 count_dict 에 정리
"""
def data_check(path ="./dataset/YogaPoses"):
    count_dict = {}
    for root, dirs, files in os.walk(path):
        if files != [] and str(root.split("\\")[-1]) in classes:
            count_dict[str(root.split('\\')[-1])] = len(files)

    return count_dict

# counts = data_check()
# print(counts)
# plt.bar(list(counts.keys()), list(counts.values()))
# plt.show()


# 2. 데이터 train valid
"""
각 폴더 안의 파일 갯수의 10%를 validation data으로 설정하고 나머지 90%를 train data으로 설정한다.
랜덤하게 나누기위해 np.random.randint로 0부터 총 파일 갯수 중 총 파일수의 10%를 size로 설정해서 랜덤한 숫자를 추출한다.
dictionary로 정리하기위해 key는 label의 요소가 될 수 있게 split하고, values는 key에 해당하는 폴더 안의 파일의 경로들이 된다.
"""
def data_split(path = "./dataset/YogaPoses", split_predictions = 0.1):
    train_dict  = {}
    val_dict    = {}
    counts      = data_check(path)
    for root, dirs, files in os.walk(path):
        if files != [] and str(root.split('\\')[-1]) in classes:
            file_paths = [os.path.join(root, files[i]) for i in range(len(files))]  # list - file paths 정리
            valid_idx = np.random.randint(low = 0, high = len(files), size=int(len(files)*split_predictions))   # 0부터 총 파일 갯수만큼의 index 중 10%를 랜덤하게 추출해서 valid_idx 설정
            train_idx = list(set(range(0, len(files))) - set(valid_idx))    # 0부터 총 파일 갯수의 index 중 valid_idx를 뺀 나머지를 train_idx 설정
            train_dict[str(root.split('\\')[-1])] = [file_paths[idx] for idx in train_idx]
            val_dict[str(root.split('\\')[-1])] = [file_paths[idx] for idx in valid_idx]

    return train_dict, val_dict

"""
train, validation split data를 label별로 갯수를 확인할 수 있다.
"""
train_split_data, val_split_data = data_split()
print("Train data size : ", [len(l) for l in train_split_data.values()])
print("Val data size : ", [len(l) for l in val_split_data.values()])


# 3. custom dataset
"""
init : dictionary로 정리한 train_split_data와 val_split_data와 transform을 dataset에 전달
getitem : transform한 이미지와 label을 반환
len : data의 values() 갯수의 합
"""
class YogaPosesData(Dataset):   # pytorch Dataset

    # chess Piece data class
    def __init__(self, data_dict, transform=None):
        # Args : data_dict(dict)
        self.data_dict = data_dict
        self.transform = transform

    def __getitem__(self, idx):
        counts = [len(l) for l in self.data_dict.values()]  # 178, 181, 178, 180, 179
        sum_counts = list(np.cumsum(counts))                # cumsum : 원소들의 누적 합
        sum_counts = [0] + sum_counts + [np.inf]

        for c, v in enumerate(sum_counts):
            if idx < v:
                i = (idx - sum_counts[c - 1]) - 1
                break

        label = list(self.data_dict.keys())[c-1]
        img = Image.open(self.data_dict[str(label)][i]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, classes.index(str(label))

    def __len__(self):
        return sum((len(l) for l in self.data_dict.values()))


# 4. data augmentation
train_data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomVerticalFlip(),    # 50%의 확률로 이미지를 수직으로 뒤집음
    transforms.RandomAdjustSharpness(sharpness_factor=1.5), # 50%의 확률로 선명도 1.5배 증가
    transforms.RandomHorizontalFlip(),  # 50%의 확률로 이미지 좌우 뒤집음
    transforms.ColorJitter(),           # 이미지 밝기, 대비, 채도 및 색조를 무작위 변경
    transforms.ToTensor()
])

val_data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(),
    transforms.ToTensor()
])


# 5. dataset
data_train = YogaPosesData(train_split_data, transform = train_data_transform)
data_valid = YogaPosesData(val_split_data, transform = val_data_transform)


# 6. train data, val data check
t_idx = np.random.randint(0, len(data_train))
v_idx = np.random.randint(0, len(data_valid))

print("Total number a train images >> ", len(data_train))
print("Total number a val images >> ", len(data_valid))

t_img, t_label = data_train[t_idx]
v_img, v_label = data_valid[v_idx]

# show train image check
# plt.figure(figsize=(8, 5))
# plt.subplot(121)
# plt.imshow(t_img.numpy().transpose(1, 2, 0))
# plt.title(f"Train Data class = {classes[t_label]}")
#
# plt.subplot(122)
# plt.imshow(v_img.numpy().transpose(1, 2, 0))
# plt.title(f"Valid Data class = {classes[v_label]}")
# plt.show()


# data loader
train_loader = DataLoader(data_train, batch_size=50, shuffle=True)
valid_loader = DataLoader(data_valid, batch_size=50, shuffle=False)

# loss function
criterion = nn.CrossEntropyLoss()

# device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
print("device info >> ", device)

# model
def base_model_build():
    # load the pretrained
    vgg11 = models.vgg11(pretrained=True)
    # print(vgg11)
    for param in vgg11.features.parameters():
        param.requires_grad = False
    n_inputs = vgg11.classifier[6].in_features
    last_layer = nn.Linear(n_inputs, len(classes))
    # 4096, 5 // len(classes)가 5개로 들어감
    # Linear(in_features=4096, out_features=5, bias=True)

    vgg11.classifier[6] = last_layer
    print(vgg11)

    if device:
        print("training...")
        vgg11.to(device)

    return vgg11

# loss, accuracy
def loss_acc_visualize(history, optim, path):
    plt.figure(figsize=(20, 10))
    plt.suptitle(str(optim))
    plt.subplot(121)
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend()
    plt.title("Loss Curvers")

    plt.subplot(122)
    plt.plot(history['train_acc'], label='train_acc')
    plt.plot(history['val_acc'], label='val_acc')
    plt.legend()
    plt.title("Acc Curvers")

    plt.savefig(str(path) + 'loss_acc.png')

# gradient
def grad_visualize(history, optim, path, ylimit=10):
    # gradient norm distribution
    plt.figure(figsize=(20, 10))
    plt.suptitle(str(optim))
    plt.subplot(131)
    sns.kdeplot(weight_grads1, shade=True)
    sns.kdeplot(bias_grads1, shade=True)
    plt.legend(['weight', 'bias'])
    plt.title("Linear layer 1")
    plt.ylim(0, ylimit)

    plt.subplot(132)
    sns.kdeplot(weight_grads2, shade=True)
    sns.kdeplot(bias_grads2, shade=True)
    plt.legend(['weight', 'bias'])
    plt.title("Linear layer 2")
    plt.ylim(0, ylimit)

    plt.subplot(133)
    sns.kdeplot(weight_grads3, shade=True)
    sns.kdeplot(bias_grads3, shade=True)
    plt.legend(['weight', 'bias'])
    plt.title("Linear layer 3")
    plt.ylim(0, ylimit)

    plt.savefig(str(path) + "grad_norms.png")

# predict
def visual_predict(model, data=data_valid):
    c = np.random.randint(0, len(data))
    img, label = data[c]

    with torch.no_grad(): # 학습을 진행하지 않고 validation을 보겠다
        model.eval()
        out = model(img.view(1, 3, 224, 224).to(device))
        out = torch.exp(out)
        print(out)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(img.numpy().transpose(1, 2, 0))
    plt.title(str(classes[label]))

    plt.subplot(122)
    plt.barh(classes, out.cpu().numpy()[0])

    plt.show()

def class_accuracies(model, data_dict=val_split_data, classes=classes):
    accuracy_dic = {}
    with torch.no_grad():
         model.eval()

         for c in data_dict.keys():
             correct_count=0
             total_count = len(data_dict[str(c)])
             gt = classes.index(str(c))

             for path in data_dict[str(c)]:
                 im = Image.open(path).convert('RGB')

                 im = transforms.ToTensor()(im)
                 im = transforms.Resize((224, 224))(im)
                 out = model(im.view(1, 3, 224, 224)).to(device)
                 out = torch.exp(out)
                 pred = list(out.cpu.numpy()[0])
                 pred = pred.index(max(pred))

                 if gt == pred:
                     correct_count += 1

             print(f"Acc for class {str(c)} : ", correct_count / total_count)
             accuracy_dic[str(c)] = correct_count / total_count

    return accuracy_dic