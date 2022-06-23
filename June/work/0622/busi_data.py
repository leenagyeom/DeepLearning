import cv2
import os
import pydicom
import glob
from PIL import Image

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torch.nn as nn
from tqdm import tqdm
import torchvision.transforms.functional as TF

device = "cuda" if torch.cuda.is_available() else "cpu"
data_path = "../../22_dicom/Dataset_BUSI_with_GT"
data_dir = os.listdir(data_path)

files = []  # save all images
labels = [] # set for each images

# read file

for folder in data_dir:
    fileList = glob.glob(os.path.join(data_path, folder, "*"))
    labels.extend([folder for l in fileList])
    files.extend(fileList)

# print(len(files), len(labels))

# create two list to hold only non-masking filter image and labels for each on
selected_files = []
selected_labels = []

for file, label in zip(files, labels):
    if 'mask' not in file:
        selected_files.append(file)
        selected_labels.append(label)
# print(len(selected_files), len(selected_labels))

images = {
    'image' : [],
    'target' : [],
}

print("Preparing the image..")

for i, (file, label) in enumerate(zip(selected_files, selected_labels)):
    images["image"].append(file)
    images["target"].append(label)


x_train, x_test, y_train, y_test = train_test_split(images["image"], images["target"], test_size=0.1)

class MyCustomData(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        data = self.x[index]
        label = self.y[index]

        image = Image.open(data).convert("RGB")

        label_temp = 0
        if label == "benign": label_temp = 0
        elif label == "malignant": label_temp = 1
        elif label == "normal": label_temp = 2

        if self.transform is not None:
            image = self.transform(image)

        return image, label_temp

    def __len__(self):
        return len(self.x)


image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAdjustSharpness(1.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
])


def train(num_epoch, model, train_loader, test_loader, criterion, optimizer,save_dir, val_every, device):

    print("String... train !!! ")
    best_loss = 9999
    for epoch in range(num_epoch):
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)

            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, argmax = torch.max(output, 1)
            acc = (labels == argmax).float().mean()

            print("Epoch [{}/{}], Step [{}/{}], Loss : {:.4f}, Acc : {:.2f}%".format(
                epoch + 1, num_epoch, i +
                1, len(train_loader), loss.item(), acc.item() * 100
            ))

            if (epoch + 1) % val_every == 0:
                avg_loss = validation(
                    epoch + 1, model, test_loader, criterion, device)
                if avg_loss < best_loss:
                    print("Best prediction at epoch : {} ".format(epoch + 1))
                    print("Save model in", save_dir)
                    best_loss = avg_loss
                    save_model(model, save_dir)

    save_model(model, save_dir, file_name="last.pt")


def validation(epoch, model, test_loader, criterion, device):
    print("Start validation # {}".format(epoch))
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        total_loss = 0
        cnt = 0
        for i, (imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total += imgs.size(0)
            _, argmax = torch.max(outputs, 1)
            correct += (labels == argmax).sum().item()
            total_loss += loss
            cnt += 1
        avg_loss = total_loss / cnt
        print("Validation # {} Acc : {:.2f}% Average Loss : {:.4f}%".format(
            epoch, correct / total * 100, avg_loss
        ))

    model.train()
    return avg_loss


def save_model(model, save_dir, file_name="best.pt"):
    output_path = os.path.join(save_dir, file_name)
    torch.save(model.state_dict(), output_path)


def eval(model, test_loader, device):
    print("Starting evaluation")
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for i, (imgs, labels) in tqdm(enumerate(test_loader)):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            # 점수가 가장 높은 클래스 선택
            _, argmax = torch.max(outputs, 1)
            total += imgs.size(0)
            correct += (labels == argmax).sum().item()

        print("Test acc for image : {} ACC : {:.2f}".format(
            total, correct / total * 100))
        print("End test.. ")


def get_model(n_classes, image_channels=3):
    model = resnet50(pretrained=True)
    for p in model.parameters():
        p.requires_grad = True
    inft = model.fc.in_features
    model.fc = nn.Linear(in_features=inft, out_features=n_classes)
    model.conv1 = nn.Conv2d(
        image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    return model



# data
train_data = MyCustomData(x_train, y_train, transform=image_transform)
valid_data = MyCustomData(x_test, y_test, transform=image_transform)
# for i in train_data:
#     image, label = i
#     print(image, label)
#     pass


# dataloader
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=16, shuffle=False)


# model prepare
model = get_model(3)
model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss().to(device)


# etc
os.makedirs("./weights", exist_ok=True)
os.makedirs("./results", exist_ok=True)

save_weights_dir = "./weights"
save_results_dir = "./results"
val_every = 1
num_epochs = 30


# eval 결과
# model.load_state_dict(torch.load(os.path.join(save_results_dir,"./last.pt")))

if __name__ == "__main__":
    train(num_epochs, model, train_loader, valid_loader, criterion, optimizer, save_results_dir, val_every, device)

    # eval(model, valid_loader, device)
