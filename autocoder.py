import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from torch.utils.tensorboard import SummaryWriter


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
            # torch.nn.Dropout(0.5),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 64),
            # torch.nn.Dropout(0.5),
            torch.nn.Tanh(),        
            torch.nn.Linear(64, 32),
            # torch.nn.Dropout(0.3),            
            torch.nn.Tanh(),
            torch.nn.Linear(32, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, out_features)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(3, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 32),
            # torch.nn.Dropout(0.3),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 64),
            # torch.nn.Dropout(0.5),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 128),
            # torch.nn.Dropout(0.5),
            torch.nn.Tanh(),
            torch.nn.Linear(128, in_features),
            torch.nn.Sigmoid()
        )
    def forward(self,x):
        en = self.encoder(x)
        de = self.decoder(en)
        return en, de

writer = SummaryWriter()
  
net = AutoCoder(28*28,3)
# writer.add_graph(net)

opt = torch.optim.Adam(net.parameters(),lr=LR)
loss_function = torch.nn.MSELoss()
lr_sh = torch.optim.lr_scheduler.StepLR(opt, step_size=2000,gamma=0.9)

f, axs = plt.subplots(2, BANTCH_SIZE, figsize=(5, 2))
for step,(bantch_x, bantch_y) in enumerate(train_data_loader):
    # 三维数据变成二维
    in_data = bantch_x.view(-1, 28*28)
    target_data = bantch_x.view(-1, 28*28)
    en,de = net(in_data)
    loss = loss_function(de, target_data)
    
    writer.add_scalars('loss', {'loss': loss}, step)
    
    opt.zero_grad()
    loss.backward()
    opt.step()
    lr_sh.step()
    if step % 500 == 0:
        # 将模型切换为验证模式        
        net.eval()
        en,de = net(in_data)
        # 将模型切换为训练模式
        net.train()       
        # 将一组训练和验证数据进行形状变换，一组变成一张图片纵向排列
        tmp_images = torch.cat([in_data.view(-1, 28,28).data, de.view(-1, 28,28).data],dim=0).view(-1,1,28,28)
        # 将纵向排列变换为横向排列,但是单个角度也会随之变换
        # tmp_images = tmp_images.transpose(3, 2)
        # 这种将图片合成一张比较好，当然也可以通过形状变换实现相同的排列方式stack可以实现
        tmp_images = torchvision.utils.make_grid(tmp_images,nrow=5)
        writer.add_image('loss/image', tmp_images, global_step=step) 
        for i in range(BANTCH_SIZE):            
            axs[0][i].clear()
            axs[1][i].clear()
            axs[0][i].imshow(in_data[i,:].view(28,28).data)
            axs[1][i].imshow(de[i,:].view(28,28).data)
            plt.suptitle('step: {}, loss: {}'.format(step, loss.data))
        # plt.draw()
        # plt.pause(0.1)
        
        # 这个每个figure都会产生一个文件，有点费文件夹
        # writer.add_figure('loss/figure', f,global_step=step)
        print(opt.param_groups[0]['lr'])
        
    writer.close()
        


    
        



    
