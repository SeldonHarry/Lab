import torch
import torchvision
import torch.nn.functional as F
import Models as md
import EYE as eye
import random

device=torch.device('cuda:0')
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def accuracy(y_hat, y):
    astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
    reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))

def train_epochs_gpu(net,train_iter,test_iter,loss,optim):
    """使用cuda:0作为device训练,使用前需配置好环境"""
    if isinstance(net,torch.nn.Module):
        net.train()
    metric=Accumulator(3)
    for X,y in train_iter:
        X,y=X.to(device),y.to(device)
        y_hat=net(X)
        l=loss(y_hat,y)
        if isinstance(optim,torch.optim.Optimizer):
            optim.zero_grad()
            l.mean().backward()
            optim.step()
        metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())
    test_acc=Accumulator(2)
    if isinstance(net,torch.nn.Module):
        net.eval()
    for X,y in test_iter:
        X,y=X.to(device),y.to(device)
        y_hat=net(X)
        test_acc.add(accuracy(y_hat,y),y.numel())
    print(f"Train loss: {metric[0]/metric[2]:.4f}\tTrain acc: {metric[1]/metric[2]:.4f}")
    print(f"Test acc: {test_acc[0]/test_acc[1]:.3f}")
    return [test_acc[0]/test_acc[1],metric[1]/metric[2]]


def FGSM(image,epsilon,grad):
    signed_data_grad=grad.sign()
    perturbed_image=image+epsilon*signed_data_grad
    perturbed_image=torch.clamp(perturbed_image,0,1)#由于至现在我们的模型还没有进行压缩映射
    return perturbed_image

def noise(image,epsilon):#用来产生噪声形成对照组，其中噪声的幅值也用EPSILON来限定
    #noises=torch.abs(torch.rand_like(image)*epsilon)
    noises=torch.rand_like(image)*epsilon*2
    return torch.clamp(image+noises,0,1)

def attack(net,test_iter,epsilon,model_name=None):
    """用于三个模型之间的Transferability验证"""
    print(f"Generate Model Name:{model_name}")
    std,noisynorm,metric1,metirc2=[],[],[[] for x in range(len(epsilon))],[[] for x in range(len(epsilon))]
    idx=0
    for _ in epsilon:
        print(f"With Epsilon equals {_}:")
        device=torch.device('cuda:0')
        Resize=torchvision.transforms.Resize(224)
        MLP=md.MLP(0.15,load_dict=True).to(device)
        LeNet=md.LeNet(load_dict=True).to(device)
        VGG=md.VGG(load_dict=True).to(device)
        if isinstance(net,torch.nn.Module):
            net.eval()
        perturbed4MLP,perturbed4LeNet,perturbed4VGG=Accumulator(2),Accumulator(2),Accumulator(2)
        noisy4MLP,noisy4LeNet,noisy4VGG=Accumulator(2),Accumulator(2),Accumulator(2)
        for X,y in test_iter:
            X,y=X.to(device),y.to(device)
            X.requires_grad=True
            y_hat=net(X)
            if torch.argmax(y_hat,dim=1)!=y:
                continue
            #perturbed是在之前标签识别正确的基础上进行干扰的
            
            L=F.cross_entropy(y_hat,y,reduction="mean")
            net.zero_grad()
            L.backward()
            X_grad=X.grad.data
            flag=X.numel()==28*28
            if flag==False:
                pytorch_resize=torchvision.transforms.Resize(28)
                noisyX=noise(pytorch_resize(X),_)
                X_0=FGSM(X.reshape(224,224),_,X_grad)
                noisyX_0=noise(X.reshape(224,224),_)
                X=pytorch_resize(X_0)

                perturbed_data4MLP=X.reshape(1,28,28).to(device)
                perturbed_data4LeNet=X.reshape(1,1,28,28).to(device)
                perturbed_data4VGG=X_0.reshape(1,1,224,224).to(device)
            else:
                noisyX=noise(X.reshape(28,28),_)
                noisyX_0=noise(Resize(X),_)
                X=FGSM(X.reshape(28,28),_,X_grad)
                X_0=Resize(X)
                perturbed_data4MLP=X.reshape(1,28,28).to(device)
                perturbed_data4LeNet=X.reshape(1,1,28,28).to(device)
                perturbed_data4VGG=X_0.reshape(1,1,224,224).to(device)

            predict4MLP=MLP(perturbed_data4MLP)
            predict4LeNet=LeNet(perturbed_data4LeNet)
            predict4VGG=VGG(perturbed_data4VGG)

            noisyPred4MLP=MLP(noisyX.reshape(1,28,28).to(device))
            noisyPred4LeNet=LeNet(noisyX.reshape(1,1,28,28).to(device))
            noisyPredVGG=VGG(noisyX_0.reshape(1,1,224,224).to(device))

            perturbed4MLP.add(accuracy(predict4MLP,y),y.numel())
            perturbed4LeNet.add(accuracy(predict4LeNet,y),y.numel())
            perturbed4VGG.add(accuracy(predict4VGG,y),y.numel())

            noisy4MLP.add(accuracy(noisyPred4MLP,y),y.numel())
            noisy4LeNet.add(accuracy(noisyPred4LeNet,y),y.numel())
            noisy4VGG.add(accuracy(noisyPredVGG,y),y.numel())
        ls1=[perturbed4MLP[0]/perturbed4MLP[1],perturbed4LeNet[0]/perturbed4LeNet[1],perturbed4VGG[0]/perturbed4VGG[1]]
        ls2=[noisy4MLP[0]/noisy4MLP[1],noisy4LeNet[0]/noisy4LeNet[1],noisy4VGG[0]/noisy4VGG[1]]
        metric1[idx],metirc2[idx]=ls1,ls2

        if _==0:
            for x in range(3):
                std.insert(x,ls1[x])
                noisynorm.insert(x,ls2[x])
            print(f"Standard init value: MLP: {ls1[0]:.4f}\tLeNet: {ls1[1]:.4f}\tVGG: {ls1[2]:.4f}")
        else:
            print(f"MLP Adv acc: {ls1[0]/std[0]:.4f}\nLeNet Adv acc: {ls1[1]/std[1]:.4f}\nVGG Adv acc: {ls1[2]/std[2]:.4f}")
            print(f"MLP Noisy acc: {ls2[0]/noisynorm[0]:.4f}\nLeNet Noisy acc: {ls2[1]/noisynorm[1]:.4f}\nVGG Noisy acc: {ls2[2]/noisynorm[2]:.4f}")
        print("+++===+++===+++===+++")
        idx+=1
    return metric1,metirc2
    
def test(net,test_iter,epsilon):
    """用于单个模型的对抗样本生成与测试"""
    if isinstance(net,torch.nn.Module):
        net.eval()
    cnt,idx=0,random.randint(1,1000)
    clean,perturbed,noises=Accumulator(2),Accumulator(2),Accumulator(2)
    for X,y in test_iter:
        cnt+=1
        X,y=X.to(device),y.to(device)
        flag=X.numel()==28*28
        size=28 if flag else 224
        X.requires_grad=True
        y_hat=net(X)
        clean.add(accuracy(y_hat,y),y.numel())
        if torch.argmax(y_hat,dim=1)!=y:
            continue
        L=F.cross_entropy(y_hat,y,reduction="mean")
        net.zero_grad()
        L.backward()
        X_grad=X.grad.data
        perturbed_data=FGSM(X.reshape(size,size),epsilon,X_grad).reshape(1,1,size,size).to(device)
        predict=net(perturbed_data)
        perturbed.add(accuracy(predict,y),y.numel())
        X_noise=noise(X.reshape(size,size),epsilon).reshape(1,1,size,size).to(device)#打完断点查看，noise tensor和x grad tensor的元素数量级相近
        comp=net(X_noise)
        noises.add(accuracy(comp,y),y.numel())
        if cnt==idx:
            resize=torchvision.transforms.Resize(28)
            ls=[resize(X).reshape(28,28).detach().cpu().numpy(),resize(perturbed_data).reshape(28,28).detach().cpu().numpy(),resize(X_noise).reshape(28,28).detach().cpu().numpy()]
            #eye.checkout_fig(ls)
    if epsilon==0:
        print(f"Test acc: {clean[0]/clean[1]:.4f}\nAdv acc: {perturbed[0]/perturbed[1]:.4f}\nNoise acc: {noises[0]/noises[1]:.4f}")
    else:
        print(f"Adv acc: {perturbed[0]/perturbed[1]:.4f}\nNoise acc: {noises[0]/noises[1]:.4f}")
    return [noises[0]/noises[1],perturbed[0]/perturbed[1]],ls