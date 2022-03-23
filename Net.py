import torch
from  torch import nn
#MLP网络，-以及CNN，根据参数合并创建
class Net(nn.Module):
    def __init__(self,is_MLP=True,is_mnist=True):
        super().__init__()
        self.is_MLP=is_MLP
        if is_mnist:
            a=28*28*1
        else:
            a=3*32*32
        if self.is_MLP:

            self.layer=nn.Sequential(
                nn.Linear(a,1024),
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

        else:
            #设计CNN的网络模型
            if is_mnist:
                pg=1
                self.b = 256*18*18
            else:
                pg=3
                self.b = 256*22*22
            self.cnn_layer=nn.Sequential(
                nn.Conv2d(pg,16,3,1),
                nn.ReLU(),
                nn.Conv2d(16,32, 3, 1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, 1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, 1),
                nn.ReLU(),
                nn.Conv2d(128, 256, 3, 1),
                #nn.Softmax(dim=1)
            )
            self.MLP_layer=nn.Sequential(

                nn.Linear(self.b,128),
                nn.ReLU(),
                nn.Linear(128,64),
                nn.ReLU(),
                nn.Linear(64,10),
                nn.Softmax(dim=1)
            )


    def forward(self,x):

        if self.is_MLP:
            #返回的是MLP网络的结果
            out=self.layer(x)
            #out=self.MLP_layer(cnn_out)
            return out
        else:
            cnn_out=self.cnn_layer(x)
            cnn_out=cnn_out.reshape(-1,self.b)
            out=self.MLP_layer(cnn_out)
            return out

if __name__=="__main__":
    # net=Net(True)
    # print(net)
    net=Net(False,False)
    c=torch.randn(1,3,32,32)
    a=net.forward(c)
    print(a,a.shape)