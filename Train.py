#用于操作文件
import os.path
#用于删除文件加
import shutil
#导入神经网络
from Net import  Net
import torch
from torch import nn
#导入数据集操作包
from torch.utils.data import DataLoader
#torchvison里面有很多torch自带的数据集
from torchvision import datasets,transforms
from torch import optim
#用于tags进行one-hot处理
from torch.nn.functional import  one_hot

##为了能个使用可视化功能，需要去pip tensorboard .现在torch支持对于tensorboard的使用，需要涌入库
from torch.utils.tensorboard import SummaryWriter
#用于收集需要显示的数据的
DEVICE = "cuda"

class Train():

    #初始化定义网络模型，数据集，等
    def __init__(self,is_mlp=True,is_mnist=True):

        #开始收集数据,会自己创建logs这个文件
        self.summerWriter=SummaryWriter("logs/")
        print("文件初始化完成")
        self.is_MLP=is_mlp
        #根据用户输入，创建对应神经网络层，并转到cuda,gpu上面去
        self.net=Net(self.is_MLP,is_mnist).to(DEVICE)
        self.is_Mnist=is_mnist

        # if os.path.exists("chekpoint/1.t"):
        #     #判断8批次的wb训练参数是否存在
        #     self.net.load_state_dict(torch.load("chekpoint/1.t"))
        #     print("加载预训练权重成功")
        #从网上下载数据集
        if self.is_Mnist:
            #如果是minister，则创建mnist数据包，数据集，第一个参数，是存放地址，第二个是是否为训练集，第三个转为tensor类型，第四个是否下载
            self.MNIST_train=datasets.MNIST(root="D:\lieweidata",train=True,transform=transforms.ToTensor(),download=True)
            self.MNIST_test = datasets.MNIST(root="D:\lieweidata", train=False, transform=transforms.ToTensor(), download=False)
            #显示加载文件基本信息，总数，路径，类型因为totensor所以为tensor类型
            print(self.MNIST_train,type(self.MNIST_train))
            #data为数据具体信息，每一张账面都是照片的三维数据，形状为60000，n,v,h,w
            print(self.MNIST_train.data,self.MNIST_train.data.shape)
            #显示的是label的数据具体值，还没有进行one-hot处理，shape为60000一共60000个数据
            print(self.MNIST_train.targets,self.MNIST_train.targets.shape)
            #创建数据集
            self.MNIST_train_dataloader=DataLoader(self.MNIST_train,batch_size=300,shuffle=True)
            self.MNIST_test_dataloader=DataLoader(self.MNIST_test,batch_size=300,shuffle=True)


        else:
            #作用和上面一样，只不过这边创建的是CIFAR10的数据
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
            self.c="minist"
        else:
            self.a = self.CIFAR_train_dataloader
            self.b = self.CIFAR_test_dataloader
            self.c="cifar"
        if self.is_MLP:
            #则进行MLP算法训练

            for epoch in range(100):
                #定义训练损失总和
                MNIST_train_loss_sum=0
                for i,(imgs,tags) in enumerate(self.a):
                    #先训练
                    #在训练之前要注意，利用datasets加载的数据，结构是n, c h w，要转化一下格式才能用MLP
                    #print(imgs,imgs.shape)
                    #将imgs,tags数据加载到cuda上在GPU上运算
                    imgs=imgs.to(DEVICE)
                    tags=tags.to(DEVICE)
                    #由于MLP接收的数据为,n v 所以将C*H*W,做一下图形形状转变
                    imgs=imgs.reshape(-1,imgs.shape[2]*imgs.shape[3]*imgs.shape[1])
                    #(imgs.shape)
                    #tags本来是索引，需要onehot处理
                    #print(tags)
                    #由于tags没有最one-hot处理，做one-hot处理，形状为10
                    tags=one_hot(tags,10)
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
                #显示一轮下来训练损失平均值
                eva_Mniset_train_loss=MNIST_train_loss_sum/len(self.a)
                print(eva_Mniset_train_loss)
                #将一轮的训练损失值，写道显示板文件里面去，名称为train_loss，值为平均值，横坐标为批次
                #self.summerWriter.add_scalar("train_loss",eva_Mniset_train_loss,epoch)
                print("数据采集完毕")


                #开始进行MLP的测试部分
                MNIST_TEST_LOSS_SUM=0
                MINIST_Test_Score=0
                for i,(imgs,tags) in enumerate(self.b):
                    #训练完一次，利用测试集测试一下
                    #将数据加载到CUDA，在GPU上跑
                    imgs=imgs.to(DEVICE)
                    tags=tags.to(DEVICE)
                    #同样的，因为mlp神经网络需要N V格式数据，进行新装转换
                    imgs=imgs.reshape(-1,imgs.shape[2]*imgs.shape[3]*imgs.shape[1])
                    #tags进项ONE-HOT处理,形状为10
                    tags = one_hot(tags,10)
                    MNIST_test_out=self.net.forward(imgs)
                    #求损失平均值
                    MNIST_test_loss=torch.mean((MNIST_test_out-tags)**2)
                    #开始
                    #求损失总和
                    MNIST_TEST_LOSS_SUM+=MNIST_test_loss.item()
                    #获得最大值的索引
                    pre=torch.argmax(MNIST_test_out,dim=1)
                    #获得tags最大值的索引
                    label_tags=torch.argmax(tags,dim=1)
                    #eq是进行匹配，得到的是bool值，然后。float转为小数，最后求平均值
                    score=torch.mean(torch.eq(pre,label_tags).float())
                    MINIST_Test_Score+=score.item()
                #求精度平均值，求损失平均值
                eva_Mniset_test_score=MINIST_Test_Score/len(self.b)
                eva_MNIst_test_loss=MNIST_TEST_LOSS_SUM/len(self.b)
                #开始收集训练损失的数据，等一会用于数据可视化,参数写需要显示的数据
                self.summerWriter.add_scalars(f"mlp_all_loss_{self.c}",{"test_loss":eva_MNIst_test_loss,"train_loss":eva_Mniset_train_loss},epoch)
                self.summerWriter.add_scalar("score",eva_Mniset_test_score,epoch)
                print("epoch:",epoch,"MLP算法平均测试损失",eva_MNIst_test_loss,"平均精度",eva_Mniset_test_score)
                #保存模型参数，固定写法，文件不会自动创建，需要自己创建(保存训练好的w,b）
                #torch.save(self.net.state_dict(),f"chekpoint/{epoch}.t")
                #判断是否保存完毕
                print("批次",epoch,"参数保存完毕")



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
                    tags=one_hot(tags,10)
                    #print(tags,tags.shape)
                    cnn_train_loss=torch.mean((cnn_train_out-tags)**2)
                    #print(cnn_train_loss)

                    #清楚梯度
                    self.opt.zero_grad()
                    #求导
                    cnn_train_loss.backward()
                    #更新梯度
                    self.opt.step()
                    #cnn_loss总和
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
                    #tags做one_hot处理,形状为10，因为10分类
                    tags=one_hot(tags,10)
                    cnn_test_loss=torch.mean((cnn_test_out-tags)**2)

                    #做得分匹配操作
                    pre=torch.argmax(cnn_test_out,dim=1)
                    label_tags=torch.argmax(tags,dim=1)
                    score=torch.mean(torch.eq(pre,label_tags).float())
                    cnn_test_score_sum+=score.item()
                    cnn_test_loss_sum+=cnn_test_loss.item()
                eva_cnn_test_score=cnn_test_score_sum/len(self.b)
                eva_cnn_test_loss=cnn_test_loss_sum/len(self.b)
                self.summerWriter.add_scalars(f"cnn_loss_{self.c}",{"train_loss":cnn_train_loss,"test_loss":cnn_test_loss},epoch)
                self.summerWriter.add_scalar("cnn_score",eva_cnn_test_score,epoch)
                print("epoch",epoch,"cnn损失",eva_cnn_test_loss,"cnn精度",eva_cnn_test_score)

if __name__=="__main__":
    #第一个参数控制选择什么网络模型，True,选择，MLP，False则选择,CNN
    #第二个参数控制，需要测试的模型是MNIST，还是CIFAR，True则是MINST，False则是CIFAR
    #选择MLP，MINIST
    if os.path.exists("logs"):

        shutil.rmtree("logs")
        print("logs文件删除完成")
    #为了做数据可视化，每次程序启动的时候，我需要把之前的采集数据清空，数据在losg里面，
    train=Train(False,False)
    train()
