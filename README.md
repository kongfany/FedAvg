# Federated Learning [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4321561.svg)](https://doi.org/10.5281/zenodo.4321561)

This is partly the reproduction of the paper of [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)   
Only experiments on MNIST and CIFAR10 (both IID and non-IID) is produced by far.

Note: The scripts will be slow without the implementation of parallel computing. 

> `！！！`代码原链接[[github](https://github.com/shaoxiongji/federated-learning)]

## 文件夹结构介绍

- data：用来存放相应的数据，可以发现一开始里面是 mnist 和 cifar 的空文件夹。数据会在torchvision package中自动下载
- save：用来存放最后生成的一系列结果，比方说自己定义的 loss 和 accuracy 随通信次数的可视化显示。
- models：里面有Fed.py 用来将模型平均实现 FedAvg、Net.py 定义不同的神经网络类型如 MLP、CNN等、test.py 将最后得到的模型效果拿来测试、update.py 用来定义本地更新的内容.
- utils：里面的options.py用来存放程序的一些超参数的值，方便在程序运行过程中直接调节(不理解的可以在后面看解释)、sampling.py用来对Non-IID的数据分布进行采样的模拟(不懂的可以先放着，后面会解释)。




## Requirements
python>=3.6  
pytorch>=0.4

## Run

The MLP and CNN models are produced by:
> python [main_nn.py](main_nn.py)

Federated learning with MLP and CNN is produced by:
> python [main_fed.py](main_fed.py)

See the arguments in [options.py](utils/options.py). 

For example:
> python main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 50 --gpu 0  

`--all_clients` for averaging over all client models

NB: for CIFAR-10, `num_channels` must be 3.

## Results
### MNIST
Results are shown in Table 1 and Table 2, with the parameters C=0.1, B=10, E=5.

Table 1. results of 10 epochs training with the learning rate of 0.01

| Model     | Acc. of IID | Acc. of Non-IID|
| -----     | -----       | ----           |
| FedAVG-MLP|  94.57%     | 70.44%         |
| FedAVG-CNN|  96.59%     | 77.72%         |

Table 2. results of 50 epochs training with the learning rate of 0.01

| Model     | Acc. of IID | Acc. of Non-IID|
| -----     | -----       | ----           |
| FedAVG-MLP| 97.21%      | 93.03%         |
| FedAVG-CNN| 98.60%      | 93.81%         |


## Ackonwledgements
Acknowledgements give to [youkaichao](https://github.com/youkaichao).

## References
McMahan, Brendan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-Efficient Learning of Deep Networks from Decentralized Data. In Artificial Intelligence and Statistics (AISTATS), 2017.

## Cite As
Shaoxiong Ji. (2018, March 30). A PyTorch Implementation of Federated Learning. Zenodo. http://doi.org/10.5281/zenodo.4321561

