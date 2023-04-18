import torch
import matplotlib
matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
import numpy as np

class Net(torch.nn.Module):
    def __init__(self, in_features, hiden, out_feature):
        super().__init__()
        self.hidden = torch.nn.Linear(in_features, hiden)
        self.predition = torch.nn.Linear(hiden, out_feature)
    
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.predition(x)
        return x
    
# 输入的参数与这里指定的维度必须一致
net = Net(1, 10, 1)
print(net)
x = torch.unsqueeze(torch.linspace(-1,1,100),1)
# 生成的数据是1行100百列，即1*100。与神经网络的维度不一致，不能兼容
x1 = torch.unsqueeze(torch.linspace(-1,1,100),0)
# 这里需要将数据调整成100行一列， 100*1
x2 = torch.linspace(-1,1,100).unsqueeze(-1)
y = x.pow(2) + 0.2*torch.rand(x.size())

opt = torch.optim.SGD(net.parameters(), lr = 0.2)
loss_f = torch.nn.MSELoss()

#开启交互模式，如果开启了plt.figure()才能显示窗口
plt.ion()

for i in range(200):
    p = net(x2)
    l = loss_f(p, y)
    
    opt.zero_grad()
    l.backward()
    opt.step()
    
    if i % 5 == 0:
        # 清屏
        plt.cla()
        # 点点点
        plt.scatter(x, y)
        # 线段
        plt.plot(x.numpy(), p.data, c='red')
        # 文字
        plt.text(0.5, 0, 'loss=%.4f' % l.item())
        # 停留一会，不然看不见
        plt.pause(0.1)
        
plt.ioff()
# 没有这一步也会显示图像，这一行代码是为了阻塞程序，防止退出
plt.show()

    

