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
        # data_path info >> ./data\train\himgri\12030269234_1c3f5a8e8c_o.jpg

        data_label = data_path.split('\\')[-2]

        # 라벨만들기
        label = 0
        if "blasti" == data_label:
            label = 0
        elif "bonegl" == data_label:
            label = 1
        elif "brhkyt" == data_label:
            label = 2
        elif "cbrtsh" == data_label:
            label = 3
        elif "cmnmyn" == data_label:
            label = 4
        elif "gretit" == data_label:
            label = 5
        elif "hilpig" == data_label:
            label = 6
        elif "himbul" == data_label:
            label = 7
        elif "himgri" == data_label:
            label = 8
        elif "hsparo" == data_label:
            label = 9
        elif "indvul" == data_label:
            label = 10
        elif "jglowl" == data_label:
            label = 11
        elif "lbicrw" == data_label:
            label = 12
        elif "mgprob" == data_label:
            label = 13
        elif "rebimg" == data_label:
            label = 14
        elif "wcrsrt" == data_label:
            label = 15

        # print(data_label, label)

        images = Image.open(data_path).convert("RGB")
        if self.transform is not None:
            images = self.transform(images)
        # print(images, label)
        return images, label


    def __len__(self):
        return len(self.all_data)