import os
import glob
import numpy as np
from PIL import Image

from torch.utils.data import Dataset


CATEGORY = {'alpaca' : 0, 'not alpaca' : 1}
class MyDataSet(Dataset):

    def __init__(self, path, mode, transform=None):
        self.path = self.split_data(path, mode)
        self.mode = mode
        self.transform = transform

    def __getitem__(self, index):
        path = self.path[index]
        label_temp = path.split("\\")[0].split("/")[-1]
        label = CATEGORY[label_temp]

        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.path)

    def split_data(self, path, mode):
        data = 0
        apc_data = glob.glob(os.path.join(path,"alpaca","*.jpg"))
        not_data = glob.glob(os.path.join(path,"not alpaca","*.jpg"))
        apc_size, not_size = len(apc_data), len(not_data)
        apc_indices, not_indices = list(range(apc_size)), list(range(not_size))

        sampling_num = 0.8
        apc_split = int(np.floor(apc_size * sampling_num))
        not_split = int(np.floor(not_size * sampling_num))
        train_apc, train_not = apc_indices[:apc_split+1], not_indices[:not_split+1]
        valid_apc = [x for x in apc_indices if x not in train_apc]
        valid_not = [x for x in not_indices if x not in train_not]

        if mode == "train":
            a_data = []
            for i in train_apc:
                file = apc_data[i]
                a_data.append(file)

            n_data = []
            for i in train_not:
                file = not_data[i]
                n_data.append(file)

            data = a_data + n_data

        elif mode == "valid":
            a_data = []
            for i in valid_apc:
                file = apc_data[i]
                a_data.append(file)

            n_data = []
            for i in valid_not:
                file = not_data[i]
                n_data.append(file)

            data = a_data + n_data

        return data