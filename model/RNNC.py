import torch

class RNN(torch.nn.Module):
    def __init__(self,input_size, hidden_size):
        super(RNN, self).__init__()
        self.rnn = torch.nn.LSTM(input_size, 
                                 hidden_size, 
                                 num_layers=1, 
                                 batch_first=True)
        self.out = torch.nn.Linear(hidden_size, 10)
    
    def forward(self,x):
        out, (hc, cn) = self.rnn(x, None)
        out = self.out(out[:,-1,:])
        return out