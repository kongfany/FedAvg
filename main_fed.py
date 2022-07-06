#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# 标准库文件的调用
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
    # parse args
    # 调用utils文件夹中的option.py文件
    args = args_parser()
    #用来选择程序运行设备，如有gpu就调用gpu否则cpu
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users 数据集的下载和划分
    if args.dataset == 'mnist': #将我们设定的args传入后做判断
        # trans_mnist是用来定义我们对图片的处理方式，这里就是将图片转化为tensor类型后进行归一化操作
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        # dataset_train dataset_test的划分都是调用dataset这个库，将数据集内容下载到了我们设定存放数据的data文件夹下
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users我们通过定义不同的数据划分方式将数据分为 iid 和 non-iid 两种，用来模拟测试 FedAvg 在不同场景下的性能。返回的是一个字典类型 dict_users，key值是用户 id，values是用户拥有的图片id。
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
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

    # build model选择我们要使用的在 models 文件夹下 Nets.py 所定义过的神经网络模型
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)#net_glob 就接收返回的网络类型
    net_glob.train()#网络进行训练

    #fedavg核心代码---每个迭代轮次本地更新 --> 复制参与本轮更新的 users 的所有权重 w_locals --> 通过定义的 FedAvg 函数求模型参数的平均 --> 分发到每个用户进行更新
    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

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