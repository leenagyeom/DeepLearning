"""data -> covid, normal, Viral Pneumonia"""

"""
train
 |
 Covid
 Normal
 Viral Peneumonia 
test
 |
 Covid
 Normal
 Viral Peneumonia 
"""

import glob
import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import f1_score
from ignite.engine import Events
from ignite.metrics import Loss, Accuracy
from ignite.engine import create_supervised_evaluator, create_supervised_trainer

device = "cuda" if torch.cuda.is_available() else "cpu"

class CustomDataset(Dataset):
    def __init__(self, data_path, mode, transform=None):
        """ init : 초기값 설정 """
        """데이터 가져오기 전체 데이터 경로 불러오기"""
        self.all_data = sorted(glob.glob(os.path.join(data_path, mode, "*", "*")))
        self.transform = transform

    def __getitem__(self, index):
        data_path = self.all_data[index]
        # data_path info >>  ./Covid19-dataset/train/Covid/01.jpeg
        data_path_split = data_path.split("\\")
        labels_temp = data_path_split[2]

        label = 0
        if "Covid" == labels_temp:
            label = 0
        elif "Normal" == labels_temp:
            label = 1
        elif "Viral Pneumonia" == labels_temp:
            label = 2

        image = Image.open(data_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.all_data)


# 기울기 계산 함수, 옵티마이저, 활성함수
def resnet(n_classes, use_pretrained=True):
    model = resnet18(use_pretrained)
    for p in model.parameters():
        p.requires_grad = True
    inft = model.fc.in_features
    model.fc = nn.Linear(in_features=inft, out_features=n_classes)
    # input_size = 224
    return model

# train loop

os.makedirs('./result', exist_ok=True)

def train(trainer, train_loader, test_loader, checkpoint_path = "./result/", epochs=1):
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        iter = (trainer.state.iteration-1) % len(train_loader) + 1
        if iter % 10 == 0:
            print("Epoch[{}] Iteration[{}/{}] Loss : {:.2f}".format(
                trainer.state.epoch, iter, len(train_loader), trainer.state.output))

    @trainer.on(Events.EPOCH_COMPLETED(every=3))
    def log_training_results(trainer):
        state = evaluator.run(train_loader)
        metrics = state.metrics
        print("Training results -> Epoch : {} Avg : accuracy {:.2f}, loss {:.2f}".format(
            trainer.state.epoch, metrics["accuracy"], metrics["loss"]))
        save(model, checkpoint_path, file_name="bestmodel_{}_{}_{}.pt".format(trainer.state.epoch, metrics["accuracy"], metrics["loss"]))

    @trainer.on(Events.EPOCH_COMPLETED(every=3))
    def log_validation_results(trainer):
        # evaluate test(validation) set
        state = evaluator.run(test_loader)
        metrics = state.metrics
        print("Validation results -> Epoch : {} Avg : accuracy {:.2f}, loss {:.2f}".format(
            trainer.state.epoch, metrics["accuracy"], metrics["loss"]))

    trainer.run(train_loader, max_epochs=epochs)


# val loop
def evaluate(epoch, model, test_loader, criterion, device):
    print("Start evaluate")
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

    return correct / total * 100

# save
def save(model, save_dir, file_name="best.pt"):
    output_path = os.path.join(save_dir, file_name)
    torch.save(model.state_dict(), output_path)

image_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# dataset
train_data = CustomDataset("../../21_train/Covid19", "train", transform=image_transform)
test_data = CustomDataset("../../21_train/Covid19", "test", transform=image_transform)

# data loader
train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=6)
test_dataloader = DataLoader(test_data, batch_size=8, shuffle=True, num_workers=6)

# prepare model
model = resnet(3)
model = model.to(device)

# 하이퍼 파라미터 값
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()
validate_every = 100
checkpoint_every = 100

trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
evaluator = create_supervised_evaluator(model, device = device, metrics={'accuracy':Accuracy(), 'loss': Loss(criterion)})


if __name__ == '__main__':
    model_t = train(trainer, train_dataloader, test_dataloader, epochs=10)

    # eval
    res = evaluate(10, model, test_dataloader, criterion, device)
    print("Eval accuracy >", res)