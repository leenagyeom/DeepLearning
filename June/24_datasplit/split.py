import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split

data_path = "../work/0623/DATASET/TRAIN"    # train 데이터만 불러서 split 한다.
data_dir = os.listdir(data_path)
# print(data_dir)

for folder in data_dir:

    if folder == "R":
        file_list_R = glob.glob(os.path.join(data_path, folder, "*.jpg"))
    elif folder == "O":
        file_list_O = glob.glob(os.path.join(data_path, folder, "*.jpg"))

R_data_size = len(file_list_R)
O_data_size = len(file_list_O)
# print("R_data_size :", R_data_size, "O_data_size", O_data_size)

# 리스트 안에 요소에 숫자를 매치해서 랜덤하게 뽑아내서 경로를 가져오는 방법으로 쓴다
R_indices = list(range(R_data_size))
O_indices = list(range(O_data_size))
# print("R_data_size :", R_indices, "O_data_size", O_indices)

R_data_split_num = 0.04
O_data_split_num = 0.032
R_split = int(np.floor(R_data_size * R_data_split_num))
O_split = int(np.floor(O_data_size * O_data_split_num))
# print(R_split, O_split)

R_data_indices, O_data_indices = R_indices[:R_split+1], O_indices[:O_split+1]
# print(R_data_indices, O_data_indices)

R_data = []
for i in R_data_indices:
    path = file_list_R[i]
    R_data.append(path)

O_data = []
for i in O_data_indices:
    path = file_list_O[i]
    O_data.append(path)

all_data = R_data + O_data

# train, valid 데이터로 사용하면 됨
x_train, x_valid = train_test_split(all_data, test_size=0.2, random_state=777)
# print(len(x_train), len(x_valid))