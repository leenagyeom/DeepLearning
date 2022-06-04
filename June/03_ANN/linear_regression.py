import torch
import matplotlib.pyplot as plt

"""
데이터 정의
A 나라의 일한시간에 대한 월급 표는 다음과 같다.
일한시간      월급
   1        1500원
   2        2500원
   3        3500원
   4        4500원
   5        5500원
   
일한 시간 x, 월급 y
"""

x_list = torch.FloatTensor([1, 2, 3, 4, 5])
y_list = torch.FloatTensor([1500, 2500, 3500, 4500, 5500])

# y = wx + b
# w와 b 값을 초기화
# required_grad = True -> 학습 데이터 명시
w = torch.zeros(1, requires_grad=True)  # 1 : 1차원으로 0을 채운다.
b = torch.zeros(1, requires_grad=True)

# 경사하강법
optimizer = torch.optim.SGD([w, b], lr = 0.015)
record_cost = []

# 학습
np_epoch = 100000
for epoch in range(1, np_epoch + 1):
    h = x_list * w + b

    # loss 계산
    cost = torch.mean((h-y_list) ** 2)
    record_cost.append(cost.item())

    print("Epoch : {:4d} y = {:4f}x+{:.4f} cost {:.6f}".format(epoch, w.item(), b.item(), cost.item()))

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

plt.plot(record_cost, 'b')
plt.show()
