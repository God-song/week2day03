import torch
from torch import nn
class Net(nn.Module):
    #初始化
    def __init__(self,is_mlp,is_mnist,data):
        super(Net, self).__init__()
        self.is_mlp=is_mlp
        self.is_mnist=is_mnist
        self.pg = data[0]
        self.sp = data[0] * data[1] * data[2]
        if is_mnist:
            #pg = 1
            self.b = 256 * 20* 20
        else:
            #pg = 3
            self.b = 256 * 24 * 24
        if is_mlp:
            #构建MLP神经网络快
            self.mlp_layer=nn.Sequential(
                nn.Linear(self.sp,1024),
                nn.ReLU(),
                nn.Linear(1024,512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10),
                nn.Softmax(dim=1)
            )
        else:
            #构建cnn模型
            self.cnn_layer=nn.Sequential(
                nn.Conv2d(self.pg,32,3,1),
                nn.ReLU(),
                nn.Conv2d(32,64,3,1),
                nn.ReLU(),
                nn.Conv2d(64,128,3,1),
                nn.ReLU(),
                nn.Conv2d(128,256,3,1),
                #nn.ReLU()
            )

            self.cnn_mlp_layer=nn.Sequential(
                nn.Linear(self.b, 256),
                nn.ReLU(),
                nn.Linear(256, 10),
                nn.Softmax(dim=1)
            )

    def forward(self,x):
        if self.is_mlp:
            return self.mlp_layer(x)

        else:
            cnn_out=self.cnn_layer(x)
            cnn_out=cnn_out.reshape(-1,self.b)
            out=self.cnn_mlp_layer(cnn_out)
            return out

if __name__=="__main__":
    a = torch.randn(1,28*28)
    net=Net(False,True,[1,1,28,28])
    a=torch.randn(1,1,28,28)
    print(net.forward(a).shape)