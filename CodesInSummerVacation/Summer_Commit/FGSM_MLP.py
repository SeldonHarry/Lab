import torch
import torchvision
from torch.utils import data
import torch.nn.functional as F
from torchvision import transforms
import EYE as eye
from torch import nn
import attacker as Adv
import Models as md
import fetch_data as fd

use_pretrained = True
file_name = "./Module/MLP_MNIST_epoch14_test_acc0.9709.pth"
device = torch.device("cuda:0")

def Save_module(net, epoch, test_acc):
    file_name = ("./Module/MLP_MNIST_epoch"+ str(epoch)+ "_test_acc"+ str(float(test_acc))+ ".pth")
    torch.save(net.state_dict(), file_name)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.01)

# 定义网络并且预加载参数
dropout1 = 0.15
lr = 0.1
epsilon=[0.0,0.01,0.02,0.03,0.04,0.05,0.1,0.15,0.2,0.25]
num_epochs, batch_size = 20, 256
net_dropout = md.MLP(dropout1)

# 训练
trainer = torch.optim.SGD(net_dropout.parameters(), lr=lr)
train_iter, test_iter = fd.my_load_data_mnist(batch_size=batch_size)  # 模型加载
loss = nn.CrossEntropyLoss(reduction="none")
if use_pretrained == False:
    net_dropout.apply(init_weights)
    net_dropout.to(device)  # GPU
    param = [[], []]
    former=0.915 
    for epoch in range(num_epochs):
        print(f"In epoch {epoch+1}:")
        if epoch > 10:
            trainer = torch.optim.SGD(net_dropout.parameters(), lr=0.05)
        _ = Adv.train_epochs_gpu(net_dropout, train_iter, test_iter, loss, trainer)
        acc = _[0]  # 画图使用
        param[0].append(_[0])  # test acc
        param[1].append(_[1])  # train acc
        #储存模型的判断条件
        # if acc > 0.92 and acc == max(param[0]) and (epoch>3 and param[0][-1]-former>0.005):
        #     Save_module(net_dropout, epoch, acc)
        #     former=acc

    eye.plot_epoacc(num_epochs,param[1],param[0],name="MLP_MNIST_train_test",xlabel="Epochs",ylabel="Accuracy",)
else:
    net_dropout.to(device)
    net_dropout.load_state_dict(torch.load(file_name))
    net_dropout.eval()

param = [[], []]
fig=[[] for x in range(len(epsilon))]
cnt=0
for x in epsilon:
    print("With EPSILON equals:", x)
    _,fig[cnt]=Adv.test(net_dropout,test_iter,x)
    param[0].append(_[0])
    param[1].append(_[1])
    cnt+=1
eye.develop_photos(fig,epsilon,name="MLP_MNIST_")
eye.noise_v_perturb(epsilon, param[0], param[1], name="MLP_Adv", xlabel="Epsilon", ylabel="Accuracy")