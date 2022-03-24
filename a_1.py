import os
import shutil
import  torch
from torch import  nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from a_2 import Net
from torch import optim
from torch.nn.functional import one_hot
#用于数据可视化
from torch.utils.tensorboard import SummaryWriter
DRIVER="cuda"
class Train():
    #构建数据集
    def __init__(self,is_mlp=True,is_minst=True):
        #创建数据存储文件夹，没有会自动创建
        self.summerwriter=SummaryWriter("logs")
        self.is_mlp=is_mlp
        self.is_minist=is_minst
        if is_minst:
            #数据包为minist数据
            self.train_dataset=datasets.MNIST(root="D:\lieweidata",train=True,transform=transforms.ToTensor(),download=True)
            self.test_dataset=datasets.MNIST(root="D:\lieweidata",train=False,transform=transforms.ToTensor(),download=False)
            self.test_Dataloader=DataLoader(self.train_dataset,batch_size=300,shuffle=True)
            self.train_dataloader=DataLoader(self.test_dataset,batch_size=300,shuffle=True)
            self.dataname="MINIST"
            print("minst数据初始化成功")
        else:
            #数据包为CIFAR
            self.train_dataset=datasets.CIFAR10(root="D:\lieweidata",train=True,transform=transforms.ToTensor(),download=True)
            self.test_dataset=datasets.CIFAR10(root="D:\lieweidata",train=False,transform=transforms.ToTensor(),download=False)
            self.train_dataloader=DataLoader(self.train_dataset,batch_size=300,shuffle=True)
            self.test_Dataloader=DataLoader(self.test_dataset,batch_size=300,shuffle=True)
            self.dataname="CIFAR10"
            print("CAFIR数据初始化成功")
        #有了数据开始构建网络
        self.net=Net(is_mlp,is_minst,self.train_dataset[0][0].shape).to(DRIVER)
        #创建网络之后，优化器之前
        # if os.path.exists("chekpoint/1.t"):
        #     self.net.load_state_dict(torch.load("chekpoint/1.t"))
        #     print("数据权重加载完毕")
        self.opt = optim.Adam(self.net.parameters())
        print("网络，以及优化器初始化成功")
    def __call__(self):
        #开始训练和测试
        for epoch in range(100):
            if self.is_mlp:
                #执行MLP
                print(f"开启MLP神经网络{self.dataname}")
                train_loss_sum=0
                for i,(imgs,tags) in enumerate(self.train_dataloader):
                    #前向计算
                    #print(i,imgs.shape)
                    imgs = imgs.to(DRIVER)
                    tags = tags.to(DRIVER)
                    imgs=imgs.reshape(-1,imgs.shape[1]*imgs.shape[2]*imgs.shape[3])
                    #imgs=imgs.to(DRIVER)
                    #print("==",imgs.shape)
                    train_out=self.net.forward(imgs)
                    #对tags做one-hot处理
                    tags=one_hot(tags,10)
                    #tags=tags.to(DRIVER)
                    train_loss=torch.mean((train_out-tags)**2)
                    #清空梯度
                    self.opt.zero_grad()
                    #求导
                    train_loss.backward()
                    #更新梯度
                    self.opt.step()
                    train_loss_sum+=train_loss.item()

                eva_train_loss=train_loss_sum/len(self.train_dataloader)
                print("数据把保存成功")
                #可以保存训练好的数据
                #torch.save(self.net.state_dict(),f"chekpoint/{epoch}.t")
                print(eva_train_loss)
                #开始测试
                test_loss_sum=0
                test_score_sum=0
                for i,(imgs,tags) in enumerate(self.test_Dataloader):
                    imgs = imgs.to(DRIVER)
                    tags=tags.to(DRIVER)
                    imgs=imgs.reshape(-1,imgs.shape[1]*imgs.shape[2]*imgs.shape[3])

                    tags=one_hot(tags,10)

                    #只进行前向计算
                    test_out=self.net.forward(imgs)

                    #求损失
                    test_loss=torch.mean((test_out-tags)**2)
                    test_loss_sum+=test_loss.item()
                    #求精度
                    pre=torch.argmax(test_out,dim=1)
                    lable=torch.argmax(tags,dim=1)
                    score=torch.mean(torch.eq(pre,lable).float())
                    test_score_sum+=score.item()
                eva_test_loss=test_loss_sum/len(self.test_Dataloader)
                eva_test_score=test_score_sum/len(self.test_Dataloader)
                print("批次",epoch,"测试损失",eva_test_loss,"测试精准度",eva_test_score)
                self.summerwriter.add_scalars(f"mlp{self.dataname}",{"train_loss":eva_train_loss,"test_loss":eva_test_loss},epoch)
                self.summerwriter.add_scalar("score",eva_test_score,epoch)


            else:
                #进行cnn运算
                print(f"开启CNN神经网络{self.dataname}")
                train_loss_sum=0
                for i,(imgs,tags) in enumerate(self.train_dataloader):
                    #开始机型训练
                    #由于cnn需要的数据形状就是N C,H  W所以imgs，不需要转变
                    imgs = imgs.to(DRIVER)
                    tags = tags.to(DRIVER)
                    train_out=self.net.forward(imgs)
                    tags=one_hot(tags,10)
                    #print(train_out)
                    train_loss=torch.mean((train_out-tags)**2)
                    self.opt.zero_grad()
                    train_loss.backward()
                    self.opt.step()
                    train_loss_sum+=train_loss.item()
                eva_train_loss=train_loss_sum/len(self.train_dataloader)
                print("训练损失为",eva_train_loss)
                #训练完开始测试
                test_loss_sum=0
                test_score_sum=0
                for i,(imgs,tags) in enumerate(self.test_Dataloader):
                    #只有前向计算
                    #因为原本的数据格式就是imgs n,c,h,w，所以可以直接使用
                    imgs = imgs.to(DRIVER)
                    tags = tags.to(DRIVER)
                    test_out=self.net.forward(imgs)
                    tags=one_hot(tags,10)
                    test_loss=torch.mean((test_out-tags)**2)
                    test_loss_sum+=test_loss.item()
                    pre=torch.argmax(test_out,dim=1)
                    label=torch.argmax(tags,dim=1)
                    score=torch.mean(torch.eq(pre,label).float())
                    test_score_sum+=score.item()
                eva_test_loss=test_loss_sum/len(self.test_Dataloader)
                eva_test_score=test_score_sum/len(self.test_Dataloader)
                print("批次",epoch,"测试损失",eva_test_loss,"测试精度",eva_test_score)
                self.summerwriter.add_scalars(f"cnn{self.dataname}",{"train_loss": eva_train_loss, "test_loss": eva_test_loss}, epoch)
                self.summerwriter.add_scalar("score", eva_test_score, epoch)
if __name__=="__main__":
    if os.path.exists("logs"):
        shutil.rmtree("logs")
        print("文件删除完毕")
    train=Train(False,False)
    print(train.train_dataset[0][0].shape)
    train()
