import torch
import numpy as np
import matplotlib.pyplot as plt


TIME_STEP=10
LR=0.02

class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = torch.nn.RNN(1, 
                                32,
                                num_layers=1,       # number of rnn layer
                                # 传入的数据类型是（1，10，1），这个数据的形状是rnn定义的，人家就这么要求
                                batch_first=True    # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size))
                                )
        self.out = torch.nn.Linear(32, 1)
        
    def forward(self,x,stat):
        # rnn的返回值形状（batch，step，hidden）
        out_r, stat = self.rnn(x,stat)
        # 形状变成(10,32)，相当于(batch_size, input_size)
        out_r = out_r.view(-1,32)
        out_r = self.out(out_r)
        # 返回的结果什么形状其实自己定义就好， 接口上说明就行。 这里为了统一，也弄成和input一样的形状
        out_r = out_r.view(-1, out_r.size(0), 1)
        # 不知道为什么pytorch的rnn自己不处理state，需要手动传入
        # 如果业务上不需要这个stat，可以将其保存到类中。这里就将其返回了
        return out_r, stat

  
net = RNN()
opt = torch.optim.Adam(net.parameters(), lr=LR)
loss_fucntion = torch.nn.MSELoss()  
    
    
plt.figure(1, figsize=(12, 5))
# plt.ion()   
state = None
for step in range(60):
    start, end = step*np.pi, (step+1)*np.pi
    # data = np.linspace(start, end, TIME_STEP, dtype=np.float32, endpoint=False)
    # input_data = torch.from_numpy(np.sin(data)[np.newaxis,:,np.newaxis])
    # target_data = torch.from_numpy(np.cos(data)[np.newaxis,:,np.newaxis])
    # or
    # star-end,取TIME_STEP个
    data = torch.linspace(start, end, TIME_STEP, dtype=torch.float)
    input_data = torch.sin(data).view(1,-1, 1)
    target_data = torch.cos(data).view(1,-1, 1)
    
    
    pre_out, state = net(input_data, state)
    # 不这样做会崩溃
    state = state.data
    loss = loss_fucntion(pre_out,target_data)
    
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    plt.plot(data, target_data.view(-1).data, 'r-',label='jieguo')
    plt.plot(data, pre_out.view(-1).data, 'b-',label='pre')
    plt.draw()
    plt.pause(0.2)
    
# plt.ioff()
plt.show()

        