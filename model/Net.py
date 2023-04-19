import torch

class Net(torch.nn.Module):
    def __init__(self, in_features, hiden, out_feature):
        super().__init__()
        self.hidden = torch.nn.Linear(in_features, hiden)
        self.predition = torch.nn.Linear(hiden, out_feature)
    
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.predition(x)
        return x