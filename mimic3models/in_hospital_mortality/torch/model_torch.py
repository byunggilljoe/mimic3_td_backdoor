import torch
import torch.nn.functional as F
import torch.nn
import random
import numpy as np

class MLPRegressor(torch.nn.Module):
    def __init__(self, input_dim):
        super(MLPRegressor, self).__init__()
        self.input_dim = input_dim
        self.fc1 = torch.nn.Linear(input_dim, 400)
        self.fc2 = torch.nn.Linear(400, 2)

    def forward(self, x):
        o = F.relu(self.fc1(x))
        o = self.fc2(o)
        return o

class LogisticRegressor(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = torch.nn.Linear(input_dim, 2)

    def forward(self, x):
        o = self.fc1(x)
        return o
    

class LSTMRegressor(torch.nn.Module):
    def __init__(self, input_dim):
        super(LSTMRegressor, self).__init__()
        self.input_dim = input_dim
        self.n_layers = 2
        self.n_hidden = 16#16
        self.num_direction = 2
        assert self.num_direction in [1, 2]
        dropout_p = 0.3
        self.lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=self.n_hidden, num_layers=self.n_layers,\
                                    bias=True, batch_first=True,\
                                    dropout=dropout_p, bidirectional= True if self.num_direction == 2 else False)
        self.fc1 = torch.nn.Linear(self.n_hidden * self.num_direction, 10)
        self.fc2 = torch.nn.Linear(10, 2)
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, x):
        hidden_init = torch.zeros(self.n_layers*self.num_direction, x.size(0), self.n_hidden).cuda()
        cell_init = torch.zeros(self.n_layers*self.num_direction, x.size(0), self.n_hidden).cuda()
        hidden_and_cell = (hidden_init, cell_init)
        on, (hn, cn)=self.lstm(x, hidden_and_cell) # last output,  (last hidden, last cell)

        o = F.relu(self.dropout(self.fc1(on[:, -1, :])))
        o = self.fc2(o)
        return o
        
class CNNRegressor(torch.nn.Module):
    def __init__(self, input_dim):
        super(CNNRegressor, self).__init__()
        self.input_dim = input_dim
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.fc1 = torch.nn.Linear(3072, 32)
        self.fc2 = torch.nn.Linear(32, 2)
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = x.unsqueeze(1)
        o = F.relu(F.max_pool2d(self.conv1(x), 2))
        o = F.relu(F.max_pool2d(self.conv2(o), 2))
        o = self.flatten(o)
        o = F.relu(self.fc1(o))
        o = self.fc2(o)
        return o