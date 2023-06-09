import torch
import matplotlib
matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
import numpy as np
from model.Net import Net as SampleNet    

def train(net, count, draw_step, loss_f, opt):
    plt.figure(draw_step)
    for i in range(count):
        out = net(x)
        loss = loss_f(out, y)    
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % draw_step == 0:
            plt.cla()
            # plot and show learning process
            # 从输出数据集中按照列取最大值，取完是一个100*1的张量
            out_max = torch.max(out, 1)
            # print(out_max)
            # print(out_max.values)
            # print(out_max.indices)
            # 这里要取的是最大值对应的索引，通过序号1或者indices都可以
            # 结果要么是0，要么是1
            prediction = out_max[1]
            pred_y = prediction.data.numpy()
            target_y = y.data.numpy()
            plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=prediction, s=100, lw=0, cmap='RdYlGn')
            # 转化成numpy是为了方便计算
            preSum = (pred_y == target_y).astype(int)
            # 如果全部预测准确，应该和target_y.size一样大
            sum = preSum.sum()
            accuracy = float(sum) / float(target_y.size)
            plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
            plt.pause(0.1)

n_data = torch.ones(100, 2)
# 标准差越大曲线越平缓，标准差越小，曲线越陡峭
x0 = torch.normal(2*n_data, 1)
x1 = torch.normal(-2*n_data, 1)
# x前半段的坐标都是正的，后半段x坐标都是负的，正好映射到Y的前半段和后半段的分类（0，1）
# 所以Y作为验证标签非常合适
y0 = torch.zeros(100)
y1 = torch.ones(100)
x = torch.cat((x0,x1),0).type(torch.FloatTensor)
y = torch.cat((y0,y1),).type(torch.LongTensor)

net_quick = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2)
)
# 由于输出数据是100*2（100行2列），所以输入feature需要为2，一次取一行
net = SampleNet(2,10,2)

opt_function = torch.optim.SGD(net.parameters(), lr = 0.002)
loss_function = torch.nn.CrossEntropyLoss()


#开启交互模式，如果开启了plt.figure()才能显示窗口
plt.ion()
sampleNetName = "sampleNet.pkl"
sampleNetParameters = "sampleNetParameters.pkl"

train(net, 200, 2,loss_function,opt_function)
 
print("doing save!")
torch.save(net, sampleNetName) 
torch.save(net.state_dict(), sampleNetParameters)
print("save done!")

print("loading to net1")
net1 = torch.load(sampleNetName)
print("train net1")
train(net1, 300, 3, loss_function, opt_function)

print("loading to net2")
net2 = SampleNet(2, 10, 2)
net2.load_state_dict(torch.load(sampleNetParameters))
print("train net2")
train(net2, 400, 4, loss_function, opt_function)

plt.ioff()
# 没有这一步也会显示图像，这一行代码是为了阻塞程序，防止退出
plt.show()



# s是绘制的点的大小
# plt.scatter(x[:,0], x[:,1],c=y,s=100,linewidths=0)
# plt.show()




