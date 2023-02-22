import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        out_c = 128
        self.lstm = nn.LSTM(
            input_size=3, #压力、排量、砂浓度、方差、均值、累计液量、累计砂量 每口井
            hidden_size=out_c,
            num_layers=1,
            batch_first=True,
            # bidirectional = True,
        )
        self.bn = nn.BatchNorm1d(out_c)
        self.f = nn.LeakyReLU()
        # self.fc1 = nn.Linear(out_c, 128)
        self.fc1 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for name, param in self.lstm.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)
        # print(h_n.shape)
        # feature = h_n[1, :, :]
        feature = h_n.squeeze(0)
        # feature = feature.squeeze(0)
        # print('1',feature.shape)
        feature = self.bn(feature)
        # print('2', feature.shape)
        feature = self.f(feature)
        # print('3', feature.shape)
        feature = self.fc1(feature)
        # print('4', feature.shape)
        out = self.sigmoid(feature)
        # print('6', out.shape)
        return out
