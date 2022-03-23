import torch
a=torch.arange(300*1*32*32).reshape(300,1,32,32)
print(a.shape[1]*a.shape[2])
print(a.shape)