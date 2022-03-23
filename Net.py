import torch
from  torch import nn
#MLP网络，-以及CNN，根据参数合并创建
class Net(nn.Module):
    def __init__(self,is_MNIST=True):
        super().__init__()
        self.is_Mnist=is_MNIST
        if self.is_Mnist:

            self.layer=nn.Sequential(
                nn.Linear(28*28*1,1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512,256),
                nn.ReLU(),
                nn.Linear(256,128),
                nn.ReLU(),
                nn.Linear(128,10),
                nn.Softmax(dim=1)
                #
            )




    def forward(self,x):

        if self.is_Mnist:
            #返回的是MLP网络的结果
            out=self.layer(x)
            return out


if __name__=="__main__":
    net=Net(True)
    print(net)