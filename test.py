#!/usr/bin/env python3


import numpy as np
import torch
import torch.nn as nn
from focalloss import FocalLoss
from BCFocalLoss import BCFocalLoss


if __name__ == "__main__":

    x = torch.softmax(torch.rand(100, 2), dim=1)
    y = 1 - x[:, 1].round().long().view(100, 1)
    
    loss1 = FocalLoss(class_num=2)
    loss2 = BCFocalLoss(gamma=2, class_num=2)
    loss3 = nn.BCELoss()

    print(loss1(x, y))
    x = x[:, 1].view(100, -1)
    print(loss2(x, y))
    print(loss3(x, y.float()))

    print("# test BCELoss")


    x = torch.cat((torch.rand(50, 1) / 2, torch.rand(50, 1) / 2 + 0.5), dim=0)
    y = x.round().long()
    print(x.shape, y.shape, np.unique(y, return_counts=True))
    print(loss3(x, y.float()))
    print(loss2(x, y))
    x = torch.cat((x, x), dim=0)
    y = torch.cat((y, y), dim=0)
    print(x.shape, y.shape, np.unique(y, return_counts=True))
    print(loss3(x, y.float()))
    print(loss2(x, y))


