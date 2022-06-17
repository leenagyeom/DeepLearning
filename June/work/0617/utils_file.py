from numpy import argmax
import torchvision.transforms as transforms
import torch
import os
from tqdm import tqdm   # 프로세서 처리 바를 보여준다
import matplotlib.pyplot as plt
import configer
import pandas as pd

"""
1. augmentation
2. train loop
3. valid loop
4. save model
5. eval test code
"""

#. augmentation
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

data = []
pd_Data = 0

# 2. train loop
def train(num_epoch, model, train_loader, test_loader, criterion, optimizer,
          save_dir, val_every, device):

    print("String... train !!! ")
    best_loss = 9999
    train_loss = 0
    train_accu = 0
    test_loss = 0
    test_accu = 0
    for epoch in range(num_epoch):
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            output = model.to(device)(imgs)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, argmax = torch.max(output, 1)
            acc = (labels == argmax).float().mean()
            train_accu = acc.item()
            train_loss = loss.item()

            print("Epoch [{}/{}], Step [{}/{}], Loss : {:.4f}, Acc : {:.2f}%".format(
                epoch + 1, num_epoch, i +
                1, len(train_loader), loss.item(), acc.item() * 100
            ))

            if (epoch + 1) % val_every == 0:
                avg_accu, avg_loss = validation(epoch + 1, model, test_loader, criterion, device)
                test_accu, test_loss = avg_accu, avg_loss

                data.append([train_accu, train_loss, test_accu, test_loss])
                if avg_loss < best_loss:
                    print("Best prediction at epoch : {} ".format(epoch + 1))
                    print("Save model in", save_dir)
                    best_loss = avg_loss
                    save_model(model, save_dir)

    pd_data = pd.DataFrame(data, columns=['train_accu', 'train_loss', 'test_accu', 'test_loss'])
    loss_acc_visualize(pd_data)
    save_model(model, save_dir, file_name="last.pt")


def validation(epoch, model, test_loader, criterion, device):
    print("Start validation # {}".format(epoch))
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        total_loss = 0
        total_accu = 0
        cnt = 0
        for i, (imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model.to(device)(imgs)
            loss = criterion(outputs, labels)

            total += imgs.size(0)
            _, argmax = torch.max(outputs, 1)
            correct += (labels == argmax).sum().item()
            total_loss += loss
            cnt += 1

        avg_accu = correct / total
        avg_loss = total_loss / cnt
        print("Validation # {} Acc : {:.2f}% Average Loss : {:.4f}%".format(epoch, avg_accu * 100, avg_loss))

    model.train()
    return avg_accu, avg_loss.item()


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



def loss_acc_visualize(history, path=configer.result_path):
    plt.figure(figsize=(20, 10))

    plt.suptitle(f"SGD; {configer.lr}")

    plt.subplot(121)
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['test_loss'], label='test_loss')
    plt.legend()
    plt.title('Loss Curves')

    plt.subplot(122)
    plt.plot(history['train_accu'], label='train_accu')
    plt.plot(history['test_accu'], label='test_accu')
    plt.legend()
    plt.title('Accuracy Curves')

    plt.savefig(os.path.join(str(path),'loss_acc.png'))