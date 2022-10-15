#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    # k: model 各部分的可训练权重
    for k in w_avg.keys():
        for i in range(1, len(w)):# i: 参与训练的 clients_num
            w_avg[k] += w[i][k] # 各部分权重加和
        w_avg[k] = torch.div(w_avg[k], len(w))# 直接求平均(因为在划分数据集时, 每个客户端的私有数据量大小相同)

    return w_avg
