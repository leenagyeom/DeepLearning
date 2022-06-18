import os
import torch
import torch.nn as nn

# 모델 가중치를 저장할 폴더 생성
os.makedirs("./weights", exist_ok=True)
os.makedirs("./result", exist_ok=True)
os.makedirs("./model", exist_ok=True)

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# hyper parameters - 공통적인 것만 만들어준다
batch_size  = 36
num_epochs  = 10
nc          = 16
lr          = 0.025

save_weights_dir    = "./weights"
save_model_dir      = "./model"
data_path           = "../0617/data"
result_path         = "./result"

criterion = nn.CrossEntropyLoss()