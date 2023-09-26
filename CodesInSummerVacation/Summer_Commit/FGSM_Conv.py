import torch
import EYE as eye
from torch import nn
import attacker as Adv
import Models as md
import fetch_data as fd

use_pretrained=False#是否使用预训练参数，若否，则将开始新的训练
file_name="./Module/LeNet_MNIST_2.0_epoch16_test_acc0.9747.pth"
device=torch.device("cuda:0")

def Save_module(net,epoch,test_acc):
    file_name="./Module/LeNet_MNIST_epoch"+str(epoch)+"_test_acc"+str(float(test_acc))+".pth"
    torch.save(net.state_dict(),file_name)

def init_weights(m):
    if isinstance(m,nn.Linear or nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)

#定义网络并且预加载参数 
lr=0.9
epsilon=[0.0,0.01,0.02,0.03,0.04,0.05]
num_epochs,batch_size=30,256
net_conv=md.LeNet()

#训练
trainer=torch.optim.SGD(net_conv.parameters(),lr=lr)
train_iter,test_iter=fd.my_load_data_mnist(batch_size=batch_size)#模型加载
loss=nn.CrossEntropyLoss(reduction="none")

#选择训练模式
if use_pretrained==False:
    net_conv.apply(init_weights)
    net_conv.to(device)
    param=[[],[]]
    for epoch in range(num_epochs):
        print(f"In epoch {epoch+1}:")
        if epoch>10:
            trainer=torch.optim.SGD(net_conv.parameters(),lr=0.3)
        _=Adv.train_epochs_gpu(net_conv,train_iter,test_iter,loss,trainer)
        acc=_[0]#画图使用
        param[0].append(_[0])#test acc
        param[1].append(_[1])#train acc
        #决定是否储存模型
        # if acc>0.96 and acc==max(param[0]):
        #     Save_module(net_conv,epoch,acc)
    eye.plot_epoacc(num_epochs,param[1],param[0],name="LeNet_train_test",xlabel="Epochs",ylabel="Accuracy")#此处需做后续修改
else:
    net_conv.to(device)
    net_conv.load_state_dict(torch.load(file_name))
    net_conv.eval()

param=[[],[]]
fig=[[] for x in range(len(epsilon))]
cnt=0
for x in epsilon:
    print("With EPSILON equals:", x)
    _,fig[cnt]=Adv.test(net_conv,test_iter,x)
    param[0].append(_[0])
    param[1].append(_[1])
    cnt+=1
eye.develop_photos(fig,epsilon,name="LeNet_MNIST_")
eye.noise_v_perturb(epsilon,param[0],param[1],name="LeNet_adv",xlabel="Epsilon",ylabel="Accuracy")
