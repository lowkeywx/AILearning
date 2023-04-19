import torch
import matplotlib.pyplot as plt
from model.Net import Net
import torch.utils.data as Data

ECHOP=12
RL=0.01
BACH_SIZE=32

# 产生的书是1*1000
data_x = torch.linspace(-1,1,1000)
# 将数据转化成1000*1
data_x = torch.unsqueeze(data_x, 1)

data_x_pow = data_x.pow(2)
data_x_zeros = torch.zeros(*data_x.size())
data_x_normal = torch.normal(data_x_zeros)
data_y = data_x_pow + 0.1*data_x_normal

data_set = Data.TensorDataset(data_x,data_y)
dataloader = Data.DataLoader(data_set, BACH_SIZE, shuffle=True)

net_sgd = Net(1, 20, 1)
net_momentum = Net(1, 20, 1)
net_rmsprop = Net(1, 20, 1)
net_adam = Net(1, 20, 1)
nets = [net_sgd, net_momentum,net_rmsprop,net_adam]

loss_sgd = torch.optim.SGD(net_sgd.parameters(), lr=RL)
loss_momentum = torch.optim.SGD(net_momentum.parameters(), lr=RL, momentum=0.8)
loss_rmsprop = torch.optim.RMSprop(net_rmsprop.parameters(), lr=RL,alpha=0.9)
loss_adam = torch.optim.Adam(net_adam.parameters(),lr=RL,betas=(0.9,0.99))
opts = [loss_sgd, loss_momentum, loss_rmsprop, loss_adam]
loss_functions = torch.nn.MSELoss()

loss_rate = [[],[],[],[]]

for i in range(ECHOP):
    for step,(bach_x,bach_y) in enumerate(dataloader):
        for net,opt,l in zip(nets, opts, loss_rate):
            out = net(bach_x)
            loss = loss_functions(out, bach_y)
            
            l.append(loss.data)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
lab=['loss_sgd','loss_momentum','loss_rmsprop','loss_adam']      
for i, loss in enumerate(loss_rate):
    # 返回line对象，line对象可以设置标签等
    # 必须是三位，且不能为0，前两位分别是横向和纵向如何切割，22表示横向分成两部分，纵向分成两部分
    aix = plt.subplot(221+i)
    
    plt.plot(loss,label=lab[i])
    plt.title(lab[i])

# 没有这行将不显示标签
plt.legend(loc='upper center')
plt.ylim(0,0.3)
plt.show()
            
            

            
