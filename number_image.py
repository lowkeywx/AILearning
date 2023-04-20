import torch
import torch.utils.data as Data
import torchvision
from model.CNNTow import CNN
from model.RNNC import RNN
import matplotlib.pyplot as plt


EPOCH=1
BANTCH_SIZE=50
LR=0.2
DOWLOAD_DATA=False
DATASET_DIR='./mnist'
TEST_DATA_BANTCH=100
USE_RNN=True

train_data = torchvision.datasets.MNIST(DATASET_DIR,
                                        train=True,
                                        transform=torchvision.transforms.ToTensor(),
                                        # target_transform=torchvision.transforms.ToTensor(),
                                        download=DOWLOAD_DATA
                                        )

print(train_data.data.size())
print(train_data.targets.size())

# print(train_data.data[0])
test_data = torchvision.datasets.MNIST(DATASET_DIR,train=False)
test_data_image = test_data.test_data[:TEST_DATA_BANTCH].type(torch.FloatTensor)/255
# print(test_data_image[0])
test_data_image = torch.unsqueeze(test_data_image, 1)
if USE_RNN:
    test_data_image = test_data_image.view(-1, 28, 28)
test_data_label = test_data.targets[:TEST_DATA_BANTCH]


dataloader = Data.DataLoader(train_data,BANTCH_SIZE,shuffle=True)

# 不转换成numpy也可以显示，shape为（w，h）这可能就是不需要转换的原因
image = train_data.data[0]
# print(image.shape)
plt.figure(1,figsize=(5,5))
plt.imshow(image)
# 如果不转换成numpy则会显示类型，tensor（5）
plt.title(train_data.targets[0].numpy())
if USE_RNN:
    net = RNN(28, 64)
else:
    # 0-9,所以out_feature是10. lable应该也是10
    net = CNN(1, 10, 28, 28)
    
opt = torch.optim.Adam(net.parameters())
loss_function = torch.nn.CrossEntropyLoss()
# 这里需要设置一个新的figure，否则坐标系纵坐标会被压缩, (8,8)=(800,800)
figure2 = plt.figure(2,figsize=(8,8))
ax = figure2.add_subplot(111)

for i in range(EPOCH):
    for step, (bantch_x, bantch_y) in enumerate(dataloader):
        # 通过dataloader提取的数据是/255，转换成float了。所以，要么test_data_image手动/255. 要么也通过dataloader提取数据
        # print(bantch_x[0])
        if USE_RNN:
            bantch_x = bantch_x.view(-1, 28, 28)
        out = net(bantch_x)
        loss = loss_function(out, bantch_y)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if step % BANTCH_SIZE == 0:
            test_out = net(test_data_image)
            # 取最大值的序号
            pre_out = torch.max(test_out, 1)[1]
            value = sum(pre_out == test_data_label) / test_data_label.size(0)
            print('准确率：', value)
            
            x = torch.arange(0, TEST_DATA_BANTCH, 1)
            # 数据增加一个维度
            # pre_out = torch.unsqueeze(pre_out, 1)
            # 增加一个维度，纵坐标0-100递增，需注意cat的两个形状需要一样。可以通过size()==size()或者shape==shape进行判断
            # pre_out = torch.cat((torch.unsqueeze(torch.arange(0, 100, 1), dim=1), pre_out), dim=1)
            # lable_compare = torch.unsqueeze(test_data_label, 1)
            # lable_compare = torch.cat((torch.unsqueeze(torch.arange(0, 100, 1),dim=1), lable_compare), dim=1)
            # cla同clear
            ax.cla()
            # cla以后需要重新设置，刻度，label等
            ax.set_ylim(0,15)
            ax.set_ylabel('number')
            ax.set_xlabel('bantch')
            # x坐标可以省略
            ax.plot(x, pre_out.numpy(),'o-r', label='yuce')
            ax.plot(x, test_data_label.numpy(),'.--b', label='jieguo')
            # 应该拿出去避免重复计算，图省事就直接/2了
            ylim = ax.get_ylim()
            xlim = ax.get_xlim()
            ax.text(xlim[1]/5, ylim[1] - ylim[1] * 0.1,'step: {}, loss: {}, accuracy rate: {:.2f}'.format(step,loss, value))
            # legend必须放到plot下面
            ax.legend()
            plt.pause(0.5)
            
            