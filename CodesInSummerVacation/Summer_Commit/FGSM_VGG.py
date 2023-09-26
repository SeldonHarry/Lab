import torch
import EYE as eye
from torch import nn
import attacker as Adv
import Models as md
import fetch_data as fd

use_pretrained=True#是否使用预训练参数，若否，则将开始新的训练
file_name="./Module/VGGNet_MNIST_epoch22_test_acc0.9925.pth"
device=torch.device("cuda:0")
ratio=1#VGG网络的结构复杂度

def init_weights(m):
    if isinstance(m,nn.Linear or nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)

def Save_module(net,epoch,test_acc):
    file_name="./Module/VGGNet_MNIST_epoch"+str(epoch)+"_test_acc"+str(float(test_acc))+".pth"
    torch.save(net.state_dict(),file_name)

#定义网络并且预加载参数 以下是超参数
lr=0.02
epsilon=[0.0,0.01,0.02,0.03,0.04,0.05,0.1,0.15,0.2,0.25]#攻击强度数组
num_epochs,batch_size=30,128#训练代数与训练集打包大小

conv_arch=(((1,8),(1,16),(2,32),(2,64),(2,64)))
larger_conv_arch = [(pair[0], pair[1] *ratio) for pair in conv_arch]
net_vgg=md.VGG(conv_arch,load_dict=False)

#训练使用的函数与数据库
trainer=torch.optim.SGD(net_vgg.parameters(),lr=lr)
train_iter,test_iter=fd.my_load_data_mnist(batch_size=batch_size,resize=224)#模型加载
loss=nn.CrossEntropyLoss(reduction="none")
    
#选择训练模式
if use_pretrained==False:
    net_vgg.apply(init_weights)#预置参数
    net_vgg.to(device)   
    param=[[],[]]
    for epoch in range(num_epochs):
        print(f"In epoch {epoch+1}:")
        _=Adv.train_epochs_gpu(net_vgg,train_iter,test_iter,loss,trainer)
        acc=_[0]#画图使用
        param[0].append(_[0])#test acc
        param[1].append(_[1])#train acc
        # if acc>0.95 and acc==max(param[0]):#储存模型的判断条件
        #     Save_module(net_vgg,epoch,acc)
    eye.plot_epoacc(num_epochs,param[1],param[0],name="VGG_MNIST_train_test",xlabel="Epochs",ylabel="Accuracy")#此处需做后续修改
else:
    net_vgg.to(device)   
    net_vgg.load_state_dict(torch.load(file_name))
    net_vgg.eval()    

param=[[],[]]
fig=[[] for x in range(len(epsilon))]
cnt=0
for x in epsilon:
    print("With EPSILON equals:", x)
    _,fig[cnt]=Adv.test(net_vgg,test_iter,x)
    param[0].append(_[0])
    param[1].append(_[1])
    cnt+=1
#以下是画图开关
eye.develop_photos(fig,epsilon,name="VGG_MNIST_")
eye.noise_v_perturb(epsilon,param[0],param[1],name="VGG_adv_longeps_MNIST",xlabel="Epsilon",ylabel="Accuracy")
