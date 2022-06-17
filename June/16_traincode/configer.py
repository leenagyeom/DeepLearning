import os
import torch
import torch.nn as nn
import torch.optim as optim

# 모델 가중치를 저장할 폴더 생성
os.makedirs("./weights", exist_ok=True)
os.makedirs("./result", exist_ok=True)

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# hyper parameters - 공통적인 것만 만들어준다
batch_size  = 36
num_epochs  = 5
val_every   = 5    # 학습 돌고 평가보는 값 설정
nc          = 5
lr          = 0.025

save_weights_dir    = "./weights"
data_path           = "./dataset"
result_path         = "./result"

criterion = nn.CrossEntropyLoss().to(device)