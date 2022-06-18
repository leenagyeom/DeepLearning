import torchvision.transforms as transforms
import torch
import os
import time
import copy
import configer, dataset
from tqdm import tqdm   # 프로세서 처리 바를 보여준다
import matplotlib.pyplot as plt
import configer
import pandas as pd
import numpy as np

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
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
        ])
    }

    return data_transforms

data = []
def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로 설정
            else:
                model.eval()   # 모델을 평가 모드로 설정

            running_loss = 0.0
            running_corrects = 0

            # 데이터를 반복
            for inputs, labels, _, _ in dataloaders[phase]:
                inputs = inputs.to(configer.device)
                labels = labels.to(configer.device)

                # 매개변수 경사도를 0으로 설정
                optimizer.zero_grad()

                # 순전파
                # 학습 시에만 연산 기록을 추적
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 학습 단계인 경우 역전파 + 최적화
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 통계
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train' :
                train_loss = epoch_loss
                train_acc = epoch_acc.item()
            else :
                test_loss = epoch_loss
                test_acc = epoch_acc.item()
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 모델을 깊은 복사(deep copy)함
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        data.append([train_acc, train_loss, test_acc, test_loss])
        print()

    pd_data = pd.DataFrame(data, columns=['train_accu', 'train_loss', 'test_accu', 'test_loss'])

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # 가장 나은 모델 가중치를 불러옴
    model.load_state_dict(best_model_wts)
    return model, pd_data


def save_model(model, file_name, save_dir=configer.save_model_dir):
    output_path = os.path.join(save_dir, f"best_{file_name}.pt")
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



def loss_acc_visualize(history, modelname, path=configer.result_path):
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

    plt.savefig(os.path.join(str(path),f'loss_acc_{modelname}.png'))


def visual_predict(model, modelname, data, path=configer.result_path):
    c = np.random.randint(0, len(data))
    img, _, label, category = data[c]

    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(img.view(1, 3, 224, 224).cuda())
        out = torch.exp(out)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(img.numpy().transpose((1, 2, 0)))
    plt.title(label)
    plt.subplot(122)
    plt.barh(category, out.cpu().numpy()[0])

    plt.savefig(os.path.join(str(path),f'predict_{modelname}.png'))