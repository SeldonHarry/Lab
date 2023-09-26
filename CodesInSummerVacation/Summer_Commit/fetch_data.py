from torchvision import transforms
from torch.utils import data
import torchvision

def my_load_data_fashion_mnist(batch_size,resize=None):
    trans=[transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans=transforms.Compose(trans)#意思是按列表中的顺序进行一系列转化
    mnist_train=torchvision.datasets.FashionMNIST(
    root="D:/DataSet/Data",train=True,transform=trans,download=True
    )
    mnist_test=torchvision.datasets.FashionMNIST(
    root="D:/DataSet/Data",train=False,transform=trans,download=True
    )
    return (data.DataLoader(mnist_train,batch_size,shuffle=True),
            data.DataLoader(mnist_test,batch_size=1,shuffle=False))#test的batch大小为1

def load_data_fashion_mnist_test(resize=False):
    trans=[transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans=transforms.Compose(trans)
    mnist_test=torchvision.datasets.FashionMNIST(
    root="D:/DataSet/Data",train=False,transform=trans,download=True
    )
    return data.DataLoader(mnist_test,batch_size=1,shuffle=False)

def my_load_data_mnist(batch_size,resize=None):
    trans=[transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans=transforms.Compose(trans)#意思是按列表中的顺序进行一系列转化
    mnist_train=torchvision.datasets.MNIST(
    root="D:/DataSet/Data",train=True,transform=trans,download=True
    )
    mnist_test=torchvision.datasets.MNIST(
    root="D:/DataSet/Data",train=False,transform=trans,download=True
    )
    return (data.DataLoader(mnist_train,batch_size,shuffle=True),
            data.DataLoader(mnist_test,batch_size=1,shuffle=False))

def get_FM_labels(labels):
    text_labels=['t-shirt','trouser','pullover','dress','coat',
                 'sandal','shirt','sneaker','bag','ankle boot']
    return [text_labels[int(i)] for i in labels]

def get_MNIST_labels(labels):
    text_lables=['0','1','2','3','4','5','6','7','8','9']
    return [text_lables[int(i)] for i in labels]