import glob
import os
from PIL import Image
from torch.utils.data import Dataset

# 데이터 경로, mode, transform 인자 필요
class CustomDataset(Dataset):

    def __init__(self, path, mode, transform=None):
        # 초기값 설정, 데이터 가져오기, 경로 불러오기
        # ../14_pretrained/dataset/YogaPoses\Downdog\00000000.jpg
        self.all_data = sorted(glob.glob(os.path.join(path, mode, "*", "*.jpg")))
        self.mode = mode
        self.transform = transform


    def __getitem__(self, index):
        data_path = self.all_data[index]
        # print("data_path info >>", data_path)

        data_label = data_path.split('\\')[-2]

        # 라벨만들기
        label = 0
        if "Downdog" == data_label:
            label = 0
        elif "Goddess" == data_label:
            label = 1
        elif "Plank" == data_label:
            label = 2
        elif "Tree" == data_label:
            label = 3
        elif "Warrior2" == data_label:
            label = 4
        # print(data_label, label)

        images = Image.open(data_path).convert("RGB")
        if self.transform is not None:
            images = self.transform(images)
        # print(images, label)
        return images, label


    def __len__(self):
        return len(self.all_data)