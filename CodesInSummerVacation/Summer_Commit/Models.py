import torch
from torch import nn

def MLP(dropout,load_dict=False,File_name=None):
    net=nn.Sequential(nn.Flatten(),
                nn.Linear(784,256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256,256),
                nn.Linear(256,10))
    if load_dict==False:
        return net
    else:
        file_name="./Module/MLP_epoch10_test_acc0.8568with_ratio2.pth"if File_name==None else File_name
        net.load_state_dict(torch.load(file_name))
        return net

def LeNet(load_dict=False,File_name=None):
    net=nn.Sequential(nn.Conv2d(1,6,kernel_size=5,padding=2),
                            nn.Sigmoid(),
                            nn.AvgPool2d(kernel_size=2,stride=2),
                            nn.Conv2d(6,16,kernel_size=5,),
                            nn.Sigmoid(),
                            nn.AvgPool2d(kernel_size=2,stride=2),
                            nn.Flatten(1),
                            nn.Linear(16*5*5,120),#此之后是全连接层
                            nn.Sigmoid(),
                            nn.Linear(120,84),
                            nn.Sigmoid(),
                            nn.Linear(84,10))#LeNet
    if load_dict==False:
        return net
    else:
        file_name="./Module/LeNet_2.0_epoch19_test_acc0.8552.pth"if File_name==None else File_name
        net.load_state_dict(torch.load(file_name))
        return net
    
def VGG(conv_arch=None,load_dict=False,File_name=None):
    """"默认使用小的conv_arch 一个epoch 3060计算时间约1min"""
    def vgg_blocks(num_convs,in_channels,out_channels):#创造一组神经卷积层
        layers=[]
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
            layers.append(nn.ReLU())
            in_channels=out_channels
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        return nn.Sequential(*layers)#这个指针不会被删掉？
    def vgg(conv_arch):
        conv_blks=[]
        in_channels=1
        for (num_convs,out_channels) in conv_arch:
            conv_blks.append(vgg_blocks(num_convs,in_channels,out_channels))
            in_channels=out_channels
        return nn.Sequential(
            *conv_blks,#卷积网络
            nn.Flatten(),
            nn.Linear(out_channels*7*7,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,10)
        )
    arch=  (((1,8),(1,16),(2,32),(2,64),(2,64))) if conv_arch ==None else conv_arch
    net=vgg(arch)
    if load_dict==False:
        return net
    else:
        file_name="./Module/VGGNet_epoch18_test_acc0.9233.pth"if File_name==None else File_name
        net.load_state_dict(torch.load(file_name))
        return net
