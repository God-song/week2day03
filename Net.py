import torch
from torch import nn


# MLP网络，-以及CNN，根据参数合并创建
class Net(nn.Module):
    def __init__(self, is_MLP=True, is_mnist=True):
        super().__init__()
        self.is_MLP = is_MLP
        # 如果输入数据时MNIST则时28*28*1的形状
        # 如果时CIFIAR则是32*32*3的形状
        if is_mnist:
            a = 28 * 28 * 1
        else:
            a = 3 * 32 * 32
        # 如果是选择全连接网络则，进入if里面的语句
        if self.is_MLP:
            # 构建MLP网络模块，第一输入层数据，根据用户输入，决定A的值
            self.layer = nn.Sequential(
                nn.Linear(a, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10),
                nn.Softmax(dim=1)
                #
            )
        # 否则用户选择的是CNN模型
        else:
            # 设计CNN的网络模型

            # 如果，哦那个输入的是minist则通道为1，
            #如果用户输入CIFAR则通道为3
            if is_mnist:
                pg = 1
                self.b = 256 * 18 * 18
            else:
                pg = 3
                self.b = 256 * 22 * 22
            self.cnn_layer = nn.Sequential(
                nn.Conv2d(pg, 16, 3, 1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, 1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, 1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, 1),
                nn.ReLU(),
                nn.Conv2d(128, 256, 3, 1),
                # nn.Softmax(dim=1)
            )
            #卷积结束之后，要接一个全连接收缩数据
            #全连接的输入形状，要根据卷积输出而定
            self.MLP_layer = nn.Sequential(

                nn.Linear(self.b, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10),
                nn.Softmax(dim=1)
            )
    #构建正向函数
    def forward(self, x):
        #如果用书是MLP，则使用MLP输出
        if self.is_MLP:
            # 返回的是MLP网络的结果
            out = self.layer(x)
            # out=self.MLP_layer(cnn_out)
            return out
        #如果选择cnn,则选择cnn输出
        else:
            cnn_out = self.cnn_layer(x)
            #由于卷积结束后，输出形状为,n ,c,h,w，全连接需要接收数据为，n,v，需要形状改变
            cnn_out = cnn_out.reshape(-1, self.b)
            out = self.MLP_layer(cnn_out)
            return out


if __name__ == "__main__":
    # net=Net(True)
    # print(net)
    net = Net(False, False)
    c = torch.randn(1, 3, 32, 32)
    a = net.forward(c)
    print(a, a.shape)
