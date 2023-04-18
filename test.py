import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import Net

import matplotlib.pyplot as plt

# 准备数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# 定义模型、损失函数和优化器
model = Net(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 开始训练
num_epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
y = []
x = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        x.append(epoch)
        y.append(running_loss)
    print('Epoch %d, loss: %.3f' % (epoch+1, running_loss/len(train_loader)))

plt.scatter(x, y)
plt.show()