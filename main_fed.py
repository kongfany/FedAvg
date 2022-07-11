#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# 标准库文件的调用
import pdb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy #用于联邦学习全局模型的复制过程
import numpy as np
from torchvision import datasets, transforms
import torch

#自定义文件的调用
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img


if __name__ == '__main__':
    # 在该脚本中可以运行，但是若被引用就不会运行
    # parse args
    # 调用utils文件夹中的option.py文件
    args = args_parser()
    #用来选择程序运行设备，如有gpu就调用gpu否则cpu
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users 数据集的下载和划分
    if args.dataset == 'mnist': #将我们设定的args传入后做判断
        # 手写体数据集，0-9分类器
        # trans_mnist是用来定义我们对图片的处理方式，这里就是将图片转化为tensor类型后进行归一化操作，归一化防止数据的溢出
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        # dataset_train dataset_test的划分都是调用dataset这个库，将数据集内容下载到了我们设定存放数据的data文件夹下，transform图片处理方式
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)

        # 数据分发
        # 客户端之间数据可能是独立同分布IID，也可能是非独立同分布的。
        # sample users我们通过定义不同的数据划分方式将数据分为 iid 和 non-iid 两种，用来模拟测试 FedAvg 在不同场景下的性能。返回的是一个字典类型 dict_users，key值是用户 id，values是用户拥有的图片id。
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            # 已知是非独立同分布，将训练集以及用户的数量传递进去，进行分发
            # 将数据打乱随机分给100个人
            dict_users = mnist_noniid(dataset_train, args.num_users)
            pdb.set_trace()
            # pdb：一个断点工具。是python自带的一个包，为python提供了一种交互的源代码调试功能
            # 主要特征是包括设置断点，单步调试，进入函数模式，查看当前代码，查找栈片段，动态改变变量的值等。
    elif args.dataset == 'cifar':
        # 60000张图片的数据集
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # 模型构建
    # build model选择我们要使用的在 models 文件夹下 Nets.py 所定义过的神经网络模型
    # net_glob 就接收返回的网络类型，通过print(net_glob)可以查看具体网络结构
    # 可以在nets.py中看到包括cnn和mlp。并且在option中定义了model的默认值为mlp
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size: #1*28*28，对于输入层的定义。
            len_in *= x
        # 将相应的参数传进去，得到net_glob就是一个全局的模型，也就是server中的模型
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)#net_glob 就接收返回的网络类型
    net_glob.train()
    # 切换成训练模式，记录每一批数据的均值和方差，并进行计算更新，以此进行误差计算方向传播和参数更新
    # 一共两种模式，一种是训练模式，一种是测试模式。对应于model.eval()测试模式。

    #fedavg核心代码---每个迭代轮次本地更新 --> 复制参与本轮更新的 users 的所有权重 w_locals --> 通过定义的 FedAvg 函数求模型参数的平均 --> 分发到每个用户进行更新
    # copy weights，复制权重参数
    w_glob = net_glob.state_dict()# state_dict变量存放训练过程中需要的权重和偏置函数，w，b

    # training
    # 一些参数的定义
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients: 
        print("Aggregation over all clients")# 对所有客户端进行聚合
        w_locals = [w_glob for i in range(args.num_users)]

    for iter in range(args.epochs):# 训练每一个epoch
        loss_locals = [] # 不同client中的loss
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)#0.1*100
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # 随机选取一部分client，全部选择会增加通信量，且实验效果可能会不好
        # google输入法推荐算法，用户数量大，不能保证虽有用户在线

        # 对于每一个用户
        for idx in idxs_users:
            # 本地训练的函数
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            # 训练之后返回的值
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            # 每个迭代轮次本地更新
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            # 复制参与本轮更新的users的所有权重w和loss
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        # 通过定义的FedAvg函数求模型参数的平均
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        # state_dict一个包含module实例完整状态的字典，包含参数和缓冲区，字典的键值是参数或缓冲区的名称。W,b
        # load_state_dict是从state_dict中复制参数和缓冲区到Module及其子类中
        net_glob.load_state_dict(w_glob)

        # print loss
        #  每一次epoch都将这一次epoch整体的loss打印
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        # 一次epoch结束后将更新恍惚的模型再分发到每个用户进行更新

    # plot loss curve最后就是对模型性能结果的测试和 loss 信息可视化
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))



# Round   0, Average loss 1.580
# Round   1, Average loss 1.565
# Round   2, Average loss 1.497
# Round   3, Average loss 1.448
# Round   4, Average loss 1.595
# Round   5, Average loss 1.568
# Round   6, Average loss 1.424
# Round   7, Average loss 1.530
# Round   8, Average loss 1.613
# Round   9, Average loss 1.457
# Training accuracy: 9.00
# Testing accuracy: 9.00
