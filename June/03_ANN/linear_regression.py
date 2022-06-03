import torch
import matplotlib.pyplot as plt

# 데이터 정의

x_list = torch.FloatTensor([1, 2, 3, 4, 5])
y_list = torch.FloatTensor([1500, 2500, 3500, 4500, 5500])

# y = wx + b
# w와 b 값을 초기화
# required_grad = True -> 학습 데이터 명시
w = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 경사하강법
optimizer = torch.optim.SGD([w, b])
record_cost = []

# 학습
np_epoch = 100000
for epoch in range(1, np_epoch + 1):
    h = x_list * w + b

    # loss 계산
    cost = torch.mean((h-y_list) ** 2)
    record_cost.append(cost.item())

    print("Epoch : {:4d} y = {:4f} cost {:.6f}".format(epoch, w.item(). b.item(), cost.item()))
