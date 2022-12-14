# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimConv4(torch.nn.Module):
    def __init__(self, feature_size=64):
        super(SimConv4, self).__init__()
        self.feature_size = feature_size
        self.name = "conv4"

        self.layer1 = torch.nn.Sequential(
            nn.Conv1d(1, 8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(8),
          torch.nn.ReLU()
        )

        self.layer2 = torch.nn.Sequential(
            nn.Conv1d(8, 16, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(16),
          torch.nn.ReLU(),
        )

        self.layer3 = torch.nn.Sequential(
            nn.Conv1d(16, 32, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(32),
          torch.nn.ReLU(),
        )

        self.layer4 = torch.nn.Sequential(
            nn.Conv1d(32, 64, 4, 2, 1, bias=False),
            torch.nn.BatchNorm1d(64),
          torch.nn.ReLU(),
          torch.nn.AdaptiveAvgPool1d(1)
        )

        self.flatten = torch.nn.Flatten()

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_ = x.view(x.shape[0], 1, -1)

        h = self.layer1(x_)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)  
        h = self.flatten(h)
        h = F.normalize(h, dim=1)
        return h
