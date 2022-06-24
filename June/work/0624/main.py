import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import dataset, configs, utils, models


# set seed
utils.set_seed()

# dataset
image_transform = utils.image_transform()
train_data = dataset.MyDataSet(configs.data_path, "train", transform=image_transform["train"])
valid_data = dataset.MyDataSet(configs.data_path, "valid", transform=image_transform["valid"])

# dataloader
train_loader = DataLoader(train_data, batch_size=configs.batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=configs.batch_size, shuffle=False)

# model call
model_name = "resnet18"
net, size = models.initialize_model(model_name, num_classes=configs.nc)
net = net.to(configs.device)

# hyper parameter call - loss function, optimizer, learning schedule
criterion = configs.criterion
optimizer = optim.SGD(net.parameters(), lr=configs.lr, momentum=0.9)
lr_schedule = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

# model load
# net.load_state_dict(torch.load(os.path.join(configs.save_path, "best.pt")))

if __name__ == "__main__":
    utils.train(configs.num_epoch, net, train_loader, valid_loader, criterion, optimizer, configs.save_path, configs.val_every, configs.device)
    # utils.eval(net, valid_loader, configs.device)