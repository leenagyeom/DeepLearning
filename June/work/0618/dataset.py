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

        category = ["blasti", "bonegl", "brhkyt", "cbrtsh", "cmnmyn", "gretit", "hilpig", "himbul",
                    "himgri", "hsparo", "indvul", "jglowl", "lbicrw", "mgprob", "rebimg", "wcrsrt"]

        # 라벨만들기
        label = 0
        if category[0] == data_label:
            label = 0
        elif category[1] == data_label:
            label = 1
        elif category[2] == data_label:
            label = 2
        elif category[3] == data_label:
            label = 3
        elif category[4] == data_label:
            label = 4
        elif category[5] == data_label:
            label = 5
        elif category[6] == data_label:
            label = 6
        elif category[7] == data_label:
            label = 7
        elif category[8] == data_label:
            label = 8
        elif category[9] == data_label:
            label = 9
        elif category[10] == data_label:
            label = 10
        elif category[11] == data_label:
            label = 11
        elif category[12] == data_label:
            label = 12
        elif category[13] == data_label:
            label = 13
        elif category[14] == data_label:
            label = 14
        elif category[15] == data_label:
            label = 15

        # print(data_label, label)

        images = Image.open(data_path).convert("RGB")
        if self.transform is not None:
            images = self.transform(images)
        # print(images, label)
        return images, label, data_label, category


    def __len__(self):
        return len(self.all_data)