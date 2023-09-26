## 关于代码的一些简要介绍
+ 两个image文件夹用来储存图片，而Module文件夹则用于储存训练好的模型参数。在这份代码工程中一共有三个模型，分别用两个数据库训练(MNIST和MNIST_FASHION)。
+ 前缀为FGSM的.py文件是用于模型训练和FGSM攻击的工程代码。而位于代码文件的前三行分别用于：选择是否使用预训练数据（若选择是，则跳过训练模型的过程，直接进入FGSM攻击）、预训练参数的文件位置、训练所用的设备。此外，在代码中还有一部分超参数的设定（包括对抗攻击强度），这些超参数都经过了研究测试，使得它们尽可能地优化模型训练地过程。
```python
use_pretrained=False#是否使用预训练参数，若否，则将开始新的训练
file_name="./Module/LeNet_MNIST_2.0_epoch16_test_acc0.9747.pth"
device=torch.device("cuda:0")
```
+ EYE.py、attacker.py、Models.py是三个通用的文件，分别用于绘图、实现对抗攻击（单个模型与对抗样本在多模型之间的传递性）、建立神经网络模型。
+ fetch_data.py则用于获取训练和测试数据集，支持MNIST和MNIST_FASHION两个数据集的获取及初始化。
+ transfertest.py是用于验证对抗样本的传递性的工程代码，可以选择的参数有test_iter,net，即选择不同的测试集和网络作为对抗样本产生的来源。但是要注意更换模型时，也要更换模型加载参数。