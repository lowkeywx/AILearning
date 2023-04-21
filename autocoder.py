import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


EPOCH=1
BANTCH_SIZE=5
# 学习率不能太高，太高会造成过拟合，无法收敛
LR=0.002
DOWLOAD_DATA=False
DATASET_DIR='./mnist'

train_data = torchvision.datasets.MNIST(DATASET_DIR,
                                        train=True,
                                        transform=torchvision.transforms.ToTensor(),
                                        # target_transform=torchvision.transforms.ToTensor(),
                                        download=DOWLOAD_DATA
                                        )
train_data_loader = Data.DataLoader(train_data,BANTCH_SIZE,shuffle=True)

print(train_data.data.size())
print(train_data.targets.size())

class AutoCoder(torch.nn.Module):
    def __init__(self, in_features, out_features):
        assert(out_features < 8)
        super(AutoCoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_features, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 64),
            torch.nn.Tanh(),        
            torch.nn.Linear(64, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, out_features)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(3, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, in_features),
            torch.nn.Sigmoid()
        )
    def forward(self,x):
        en = self.encoder(x)
        de = self.decoder(en)
        return en, de

  
net = AutoCoder(28*28,3)

opt = torch.optim.Adam(net.parameters(),lr=LR)
loss_function = torch.nn.MSELoss()

f, axs = plt.subplots(2, BANTCH_SIZE, figsize=(5, 2))
for step,(bantch_x, bantch_y) in enumerate(train_data_loader):
    # 三维数据变成二维
    in_data = bantch_x.view(-1, 28*28)
    target_data = bantch_x.view(-1, 28*28)
    en,de = net(in_data)
    loss = loss_function(de, target_data)
    
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 500 == 0:
        for i in range(BANTCH_SIZE):
            axs[0][i].clear()
            axs[1][i].clear()
            axs[0][i].imshow(in_data[i,:].view(28,28).data)
            axs[1][i].imshow(de[i,:].view(28,28).data)
            plt.suptitle('step: {}, loss: {}'.format(step, loss.data))
        plt.draw()
        plt.pause(0.1)


    
        



    
