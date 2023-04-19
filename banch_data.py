import torch
import torch.utils.data as Data

data_x = torch.linspace(1,10,10)
data_y = torch.linspace(10,1,10)

dataset = Data.TensorDataset(data_x,data_y)
data_loader = Data.DataLoader(dataset, 5)
for i in range(3):
    for s, (bach_x,banch_y) in enumerate(data_loader):
        print("echo: ", i, " step: ", s, " data_x: ", bach_x, " data_y: ",banch_y)


print("==========================================================================\n")
data_loader = Data.DataLoader(dataset, 5,shuffle=True)
for i in range(3):
    for s, (bach_x,banch_y) in enumerate(data_loader):
        print("echo: ", i, " step: ", s, " data_x: ", bach_x, " data_y: ",banch_y)

