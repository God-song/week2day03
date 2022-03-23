from Net import  Net
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torch import optim
from torch.nn.functional import  one_hot
class Train():

    #初始化定义网络模型，数据集，等
    def __init__(self,is_mlp=True):
        self.is_MLP=is_mlp
        self.net=Net(self.is_MLP)


        #从网上下载数据集
        if self.is_MLP:
            #
            self.MNIST_train=datasets.MNIST(root="E:\data",train=True,transform=transforms.ToTensor(),download=True)
            self.MNIST_test = datasets.MNIST(root="E:\data", train=False, transform=transforms.ToTensor(), download=False)
            #创建数据集
            self.MNIST_train_dataloader=DataLoader(self.MNIST_train,batch_size=300,shuffle=True)
            self.MNIST_test_dataloader=DataLoader(self.MNIST_test,batch_size=300,shuffle=True)



        else:
            self.CIFAR_train=datasets.CIFAR10(root="E:\data",train=True,transform=transforms.ToTensor(),download=True)
            self.CIFAR_test=datasets.CIFAR10(root="E:\data",train=False,transform=transforms.ToTensor(),download=False)

            #创建数据集
            self.MNIST_train_dataloader=DataLoader(self.CIFAR_train,batch_size=300,shuffle=True)
            self.MNIST_test_dataloader=DataLoader(self.CIFAR_test,batch_size=300,shuffle=True)

        #创建优化器
        print("下载完成")
        self.opt=optim.Adam(self.net.parameters())

    #创建执行callback函数
    def __call__(self):
        #根据选择，选择训练MLP, or CNN
        if self.is_MLP:
            #则进行MLP算法训练
            for epoch in range(20):
                MNIST_train_loss_sum=0
                for i,(imgs,tags) in enumerate(self.MNIST_train_dataloader):
                    #先训练
                    #在训练之前要注意，利用datasets加载的数据，结构是n, c h w，要转化一下格式才能用MLP
                    imgs=imgs.reshape(-1,28*28*1)
                    #(imgs.shape)
                    #print(tags)
                    tags=one_hot(tags)
                    MNIST_train_out=self.net.forward(imgs)
                    #获得MNIST，的全连接，训练损失值
                    MNIST_train_loss=torch.mean((MNIST_train_out-tags)**2)

                    #清空梯度
                    self.opt.zero_grad()
                    #求导
                    MNIST_train_loss.backward()
                    #更新梯度
                    self.opt.step()
                    MNIST_train_loss_sum+=MNIST_train_loss.item()
                print(MNIST_train_loss_sum)
                MNIST_TEST_LOSS_SUM=0
                MINIST_Test_Score=0
                for i,(imgs,tags) in enumerate(self.MNIST_test_dataloader):
                    #训练完一次，利用测试集测试一下
                    imgs=imgs.reshape(-1,28*28*1)
                    tags = one_hot(tags)
                    MNIST_test_out=self.net.forward(imgs)
                    MNIST_test_loss=torch.mean((MNIST_test_out-tags)**2)
                    #开始
                    MNIST_TEST_LOSS_SUM+=MNIST_test_loss.item()
                    pre=torch.argmax(MNIST_test_out,dim=1)
                    label_tags=torch.argmax(tags,dim=1)
                    score=torch.mean(torch.eq(pre,label_tags).float())
                    MINIST_Test_Score+=score.item()
                eva_Mniset_test_score=MINIST_Test_Score/len(self.MNIST_test_dataloader)
                eva_MNIst_test_loss=MNIST_TEST_LOSS_SUM/len(self.MNIST_test_dataloader)

                print("epoch:",epoch,"平均测试损失",eva_MNIst_test_loss,"平均精度",eva_Mniset_test_score)




        else:
            #进行cnn训练
            pass

if __name__=="__main__":
    train=Train(True)
    train()
