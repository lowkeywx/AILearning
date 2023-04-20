import torch.nn as nn
import torch
import torch.nn.functional as F

KERNEL_SIZE=4
POOLSIZE=2

class CNN(torch.nn.modules):
    def __init__(self,in_feature,out_feature, in_w, in_h):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_feature, 16, KERNEL_SIZE),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=POOLSIZE,stride=POOLSIZE)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=KERNEL_SIZE),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=POOLSIZE,stride=POOLSIZE)
        )
        self.out = nn.Linear(32*in_w*in_w/(POOLSIZE*2)/(POOLSIZE*2), out_feature)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        # 将张量变成指定的形状，如果x是（2，3）,view(-1)将变成（6）.abs(x)
        # 简言之，将特定维数张量，变成指定维数向量，总元素数必须相等
        x = x.veiw(x.size(0), -1)
        output = self.out(x)
        return output
        