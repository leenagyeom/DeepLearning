import cv2
import os
import pydicom
import glob

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

input_data_path = "./dcm_data"
output_image_save_path = "./dcm_data_png/"
os.makedirs(output_image_save_path, exist_ok=True)

dcm_file_list = [f for f in os.listdir(input_data_path)]

for f in dcm_file_list:
    file = os.path.join(input_data_path,f)
    ds = pydicom.read_file(file)
    # print(ds)
    # print(ds["PatientID"])
    img = ds.pixel_array
    # print(img.shape)
    # exit()

    """ image save """
    # dc -> png 변경 후 라벨링
    cv2.imwrite(output_image_save_path + f.replace('dcm', 'png'), img)
