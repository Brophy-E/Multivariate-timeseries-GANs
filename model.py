"""
Created on Tue Dec 03 2019

@author: Eoin Brophy

Notebook for MV time series generation - Model
"""

import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

"""
Generator Class
---------------
This defines the Generator for evaluation. The Generator consists of two LSTM 
layers with a final fully connected layer.
"""
class Generator(nn.Module):
    def __init__(self, seq_len, batch_size, n_features=2, hidden_dim=50,
                 num_layers=2, tanh_output=False):
        super(Generator, self).__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.tanh_output = tanh_output

        self.layer1 = nn.LSTM(input_size=self.n_features, hidden_size=self.hidden_dim,
                              num_layers=self.num_layers, batch_first=True,
                              )
        if self.tanh_output == True:
            self.out = nn.Sequential(nn.Linear(self.hidden_dim, 2), nn.Tanh())
        else:
            self.out = nn.Linear(self.hidden_dim, 2)

    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, self.batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.num_layers, self.batch_size, self.hidden_dim).zero_().to(device))

        return hidden

    def forward(self, input, hidden):
        lstm_out, hidden = self.layer1(input.view(self.batch_size, self.seq_len, -1), hidden)
        lstm_out = self.out(lstm_out)

        return lstm_out


"""
Discriminator Class
---------------
This defines the Discriminator for evaluation. The Discriminator consists of 4 Conv-Pool layers, 
minibatch discrimination layer and a final fully connected layer with sigmoid activation function.
"""
class Discriminator(nn.Module):
    def __init__(self, seq_len, in_channels,
                 cv1_k, cv1_s, p1_k, p1_s,
                 cv2_k, cv2_s, p2_k, p2_s,
                 cv3_k, cv3_s, p3_k, p3_s,
                 cv4_k, cv4_s, p4_k, p4_s,
                 minibatch_layer, minibatch_init=False):
        super(Discriminator, self).__init__()
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.minibatch_init = minibatch_init
        self.minibatch_layer = int(minibatch_layer)

        self.cv1_k = cv1_k
        self.cv1_s = cv1_s
        self.cv2_k = cv2_k
        self.cv2_s = cv2_s
        self.cv3_k = cv3_k
        self.cv3_s = cv3_s
        self.cv4_k = cv4_k
        self.cv4_s = cv4_s

        self.p1_k = p1_k
        self.p1_s = p1_s
        self.p2_k = p2_k
        self.p2_s = p2_s
        self.p3_k = p3_k
        self.p3_s = p3_s
        self.p4_k = p4_k
        self.p4_s = p4_s

        cp1_out = int((((((seq_len - cv1_k) / cv1_s) + 1) - p1_k) / p1_s) + 1)
        cp2_out = int((((((cp1_out - cv2_k) / cv2_s) + 1) - p2_k) / p2_s) + 1)
        cp3_out = int((((((cp2_out - cv3_k) / cv3_s) + 1) - p3_k) / p3_s) + 1)
        cp4_out = int((((((cp3_out - cv4_k) / cv4_s) + 1) - p4_k) / p4_s) + 1)

        # The first pair of convolution-pooling layer, input size: 2*187*batch_size
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=3 * 2, kernel_size=(cv1_k, 1), stride=(cv1_s, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(p1_k, 1), stride=(p1_s, 1))
        )

        # Second convolution - pooling layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=3 * 2, out_channels=5 * 2, kernel_size=(cv2_k, 1), stride=(cv2_s, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(p2_k, 1), stride=(p2_s, 1))
        )

        # Third convolution - pooling layer
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=5 * 2, out_channels=8 * 2, kernel_size=(cv3_k, 1), stride=(cv3_s, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(p3_k, 1), stride=(p3_s, 1))
        )
        # Fourth convolution - pooling layer
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=8 * 2, out_channels=10 * 2, kernel_size=(cv4_k, 1), stride=(cv4_s, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(p4_k, 1), stride=(p4_s, 1))
        )

        if self.minibatch_layer > 0:
            self.mb = MinibatchDiscrimination(int(10 * cp4_out * 2), self.minibatch_layer, minibatch_normal_init=True,
                                              hidden_features=16)
            m = self.minibatch_layer * 1
        else:
            m = 0

        # Fully connected layer
        self.layer5 = nn.Sequential(
            nn.Linear(10 * cp4_out * 2 + m, 50),
            nn.Linear(50, 50),
            nn.Linear(50, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)

        if self.minibatch_layer > 0:
            x = self.mb(x)

        output = self.layer5(x)
        return output

"""
MinibatchDiscrimination Class
-----------------------------
This defines the MinibatchDiscrimination for use in the Discriminator. 
"""
class MinibatchDiscrimination(nn.Module):
    def __init__(self, input_features, output_features, minibatch_normal_init, hidden_features=16):
        super(MinibatchDiscrimination, self).__init__()

        self.input_features = input_features
        self.output_features = output_features
        self.hidden_features = hidden_features
        self.T = torch.nn.Parameter(torch.randn(self.input_features, self.output_features, self.hidden_features))

        if minibatch_normal_init == True:
            nn.init.normal_(self.T, 0, 1)

    def forward(self, x):
        M = torch.mm(x, self.T.view(self.input_features, -1))
        M = M.view(-1, self.output_features, self.hidden_features).unsqueeze(0)
        M_t = M.permute(1, 0, 2, 3)
        # Broadcasting reduces the matrix subtraction to the form desired in the paper
        out = torch.sum(torch.exp(-(torch.abs(M - M_t).sum(3))), dim=0) - 1

        return torch.cat([x, out], 1)
