# #!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/12 23:07
# @Author  : Wang Ziyan Yijing Liao
# @File    : main.py
# @Software: PyCharm
from torch import nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, output_size)
        # self.sfm = nn.Softmax(dim=1)
        self.sig = nn.Sigmoid()

    def forward(self, x, hs):
        # input shape: (batch_size, seq_length, num_features), and hs for hidden state
        # out:(batch_size, seq_length, hidden_size), (hn, cn)
        out, hs = self.lstm(x, hs)
        # reshape our data into the form (batches, n_hidden)
        out = out.reshape(-1, self.hidden_size)
        # input shape: (batch_size * seq_length, hidden_size)
        out = self.fc(out)
        # output shape: (batch_size * seq_length, out_size)
        # out = self.sfm(out)
        out = self.sig(out)
        return out, hs
