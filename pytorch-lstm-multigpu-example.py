import torch.nn as nn
import torch
import torch.nn.init as weight_init

class Network(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Network, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=False)
        for p in self.lstm.parameters():
            weight_init.normal_(p, std=0.1)

    def forward(self, input_var, h0):
        self.lstm.flatten_parameters()
        output, ht = self.lstm(input_var, h0)
        return output,ht

net = Network(128,256).cuda()
dp=torch.nn.DataParallel(net, dim = 1)
input_var=torch.rand(32,50,128).cuda()  #[seq_len, batch, input_size]
h0=torch.randn(2, 50,256).cuda()   #[num_layers, batch, hidden_size]
c0=torch.randn(2, 50,256).cuda()
h=(h0,c0)

out, ht=dp(input_var,h)

print("out shape: {}".format(out.shape))   #[32,50, 256]
print("ht[0] shape: {}".format(ht[0].shape))  #[2, 50 ,256]
print("ht[1] shape: {}".format(ht[1].shape))   #[2, 50, 256]

