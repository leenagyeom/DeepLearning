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




from cProfile import label
import glob
import os
from this import d
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
class CustomDataset(Dataset):
    def __init__(self, data_path, mode, transform=None):
        """ init : 초기값 설정 """
        """데이터 가져오기 전체 데이터 경로 불러오기"""
        self.all_data = sorted(
            glob.glob(os.path.join(data_path, mode, "*", "*")))
        """data_path > ./dataset/ mode -> train * -> Downdog *-> 000000.jpg """
        self.transform = transform

    def __getitem__(self, index):
        data_path = self.all_data[index]
        # data_path info >>  ./Covid19-dataset/train/Covid/01.jpeg
        print("data_path info >> ", data_path)
        data_path_split = data_path.split("/")
        print(data_path_split[3])
        labels_temp = data_path_split[3]

        label = 0
        if "Covid" == labels_temp:
            label = 0
        elif "Normal" == labels_temp:
            label = 1
        elif "Viral Pneumonia" == labels_temp:
            label = 3

        image = Image.open(data_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        print(image, label)
        return image, label

    def __len__(self):
        return len(self.all_data)


image_transform = transforms.Compose([
    transforms.ToTensor()
])

train_data = CustomDataset("./Covid19-dataset/", "train", transform=image_transform)
test_data = CustomDataset("./Covid19-dataset/", "test", transform=image_transform)


# data loader
# 기울기 계산 함수, 옵티마이저, 활성함수
# 하이퍼 파라미터 값
# train loop val loop save

"""Custom data 까지 진행 나머지"""
