from Net import  Net
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torch import optim
from torch.nn.functional import  one_hot
DEVICE = "cuda"
class Train():

    #初始化定义网络模型，数据集，等
    def __init__(self,is_mlp=True,is_mnist=True):
        self.is_MLP=is_mlp
        self.net=Net(self.is_MLP,is_mnist).to(DEVICE)
        self.is_Mnist=is_mnist


        #从网上下载数据集
        if self.is_Mnist:
            #
            self.MNIST_train=datasets.MNIST(root="D:\lieweidata",train=True,transform=transforms.ToTensor(),download=True)
            self.MNIST_test = datasets.MNIST(root="D:\lieweidata", train=False, transform=transforms.ToTensor(), download=False)
            #创建数据集
            self.MNIST_train_dataloader=DataLoader(self.MNIST_train,batch_size=300,shuffle=True)
            self.MNIST_test_dataloader=DataLoader(self.MNIST_test,batch_size=300,shuffle=True)


        else:
            self.CIFAR_train=datasets.CIFAR10(root="D:\lieweidata",train=True,transform=transforms.ToTensor(),download=True)
            self.CIFAR_test=datasets.CIFAR10(root="D:\lieweidata",train=False,transform=transforms.ToTensor(),download=False)

            #创建数据集
            self.CIFAR_train_dataloader=DataLoader(self.CIFAR_train,batch_size=300,shuffle=True)
            self.CIFAR_test_dataloader=DataLoader(self.CIFAR_test,batch_size=300,shuffle=True)

        #创建优化器
        print("下载完成")
        self.opt=optim.Adam(self.net.parameters())

    #创建执行callback函数
    def __call__(self):
        #根据选择，选择训练MLP, or CNN
        if self.is_Mnist:
            self.a = self.MNIST_train_dataloader
            self.b = self.MNIST_test_dataloader
        else:
            self.a = self.CIFAR_train_dataloader
            self.b = self.CIFAR_test_dataloader
        if self.is_MLP:
            #则进行MLP算法训练

            for epoch in range(100):
                MNIST_train_loss_sum=0
                for i,(imgs,tags) in enumerate(self.a):
                    #先训练
                    #在训练之前要注意，利用datasets加载的数据，结构是n, c h w，要转化一下格式才能用MLP
                    #print(imgs,imgs.shape)
                    imgs=imgs.to(DEVICE)
                    tags=tags.to(DEVICE)
                    imgs=imgs.reshape(-1,imgs.shape[2]*imgs.shape[3]*imgs.shape[1])
                    #(imgs.shape)
                    #tags本来是索引，需要onehot处理
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
                print(MNIST_train_loss_sum/len(self.a))
                MNIST_TEST_LOSS_SUM=0
                MINIST_Test_Score=0
                for i,(imgs,tags) in enumerate(self.b):
                    #训练完一次，利用测试集测试一下
                    imgs=imgs.to(DEVICE)
                    tags=tags.to(DEVICE)
                    imgs=imgs.reshape(-1,imgs.shape[2]*imgs.shape[3]*imgs.shape[1])
                    tags = one_hot(tags)
                    MNIST_test_out=self.net.forward(imgs).to("cuda")
                    MNIST_test_loss=torch.mean((MNIST_test_out-tags)**2)
                    #开始
                    MNIST_TEST_LOSS_SUM+=MNIST_test_loss.item()
                    pre=torch.argmax(MNIST_test_out,dim=1)
                    label_tags=torch.argmax(tags,dim=1)
                    score=torch.mean(torch.eq(pre,label_tags).float())
                    MINIST_Test_Score+=score.item()
                eva_Mniset_test_score=MINIST_Test_Score/len(self.b)
                eva_MNIst_test_loss=MNIST_TEST_LOSS_SUM/len(self.b)

                print("epoch:",epoch,"MLP算法平均测试损失",eva_MNIst_test_loss,"平均精度",eva_Mniset_test_score)




        else:
            #进行cnn训练
            for epoch in range(100):
                #循环100此完整数据集训练
                cnn_tarin_loss_sum=0
                for i,(imgs,tags) in enumerate(self.a):
                    imgs=imgs.to(DEVICE)
                    tags=tags.to(DEVICE)
                    #照片格式是N ,C H W,由于cnn算法，输入数据格式就是NCHW,所以不需要改变
                    cnn_train_out=self.net.forward(imgs).to("cuda")
                    #print(tags,tags.shape)
                    tags=one_hot(tags)
                    #print(tags,tags.shape)
                    cnn_train_loss=torch.mean((cnn_train_out-tags)**2)
                    #print(cnn_train_loss)

                    #清楚梯度
                    self.opt.zero_grad()
                    #求导
                    cnn_train_loss.backward()
                    #更新梯度
                    self.opt.step()
                    cnn_tarin_loss_sum+=cnn_train_loss.item()
                print(cnn_tarin_loss_sum/len(self.a))


                #训练完之后，开始测试
                cnn_test_loss_sum=0
                cnn_test_score_sum=0
                for i,(imgs,tags) in enumerate(self.b):
                    #测试部分之后前向步骤,卷积需要，n chw格式，现在不需要改变
                    imgs=imgs.to(DEVICE)
                    tags=tags.to(DEVICE)
                    cnn_test_out=self.net.forward(imgs)
                    #tags做one_hot处理
                    tags=one_hot(tags)
                    cnn_test_loss=torch.mean((cnn_test_out-tags)**2)

                    #做得分匹配操作
                    pre=torch.argmax(cnn_test_out,dim=1)
                    label_tags=torch.argmax(tags,dim=1)
                    score=torch.mean(torch.eq(pre,label_tags).float())
                    cnn_test_score_sum+=score.item()
                    cnn_test_loss_sum+=cnn_test_loss.item()
                eva_cnn_test_score=cnn_test_score_sum/len(self.b)
                eva_cnn_test_loss=cnn_test_loss_sum/len(self.b)
                print("epoch",epoch,"cnn损失",eva_cnn_test_loss,"cnn精度",eva_cnn_test_score)

if __name__=="__main__":
    #第一个参数控制选择什么网络模型，True,选择，MLP，False则选择,CNN
    #第二个参数控制，需要测试的模型是MNIST，还是CIFAR，True则是MINST，False则是CIFAR
    #选择MLP，MINIST
    train=Train(False,False)
    train()
