import matplotlib.pyplot as plt
import numpy as np

def plot_epoacc(epochs, train_acc, test_acc, name=None, xlabel=None, ylabel=None):
    ls=[x+1 for x in range(epochs)]
    x = np.array(ls).astype(dtype=np.str_)
    train_acc = np.array(train_acc)
    test_acc = np.array(test_acc)

    plt.rc("lines", linewidth=1.5)
    fig, ax = plt.subplots()
    (line1,) = ax.plot(x, train_acc, label="Training Accuracy")
    line1.set_dashes([3, 3, 3, 3])
    line1.set_dash_capstyle("round")

    (line2,) = ax.plot(
        x,
        test_acc,
        dashes=[4, 2.4],
        color="green",
        gapcolor="tab:pink",
        label="Test Accuracy" ,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.savefig("./image1/" + name + ".svg")
    plt.show()


def noise_v_perturb(epsilon, noise_acc, adv_acc, name=None, xlabel=None, ylabel=None):
    x = np.array(epsilon)
    noise_acc = np.array(noise_acc)
    adv_acc = np.array(adv_acc)

    plt.rc("lines", linewidth=1.5)
    plt.stem(x, noise_acc, label="Added Noise Accuracy")
    plt.plot(x, noise_acc, dashes=[4, 4])
    plt.stem(x, adv_acc, linefmt="grey", label="Added Perturbed Accuracy")
    plt.plot(x, adv_acc, color="pink", dashes=[4, 4])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig("./image1/" + name + ".svg")
    plt.show()

def transfer_test(epsilon,ls,title=None,name=None,ylabel=None,):
    x=np.array(epsilon)
    ls[0]=[1,1,1]
    MLP=np.array([ls[_][0] for _ in range(len(epsilon))])
    LeNet=np.array([ls[_][1] for _ in range(len(epsilon))])
    VGG=np.array([ls[_][2] for _ in range(len(epsilon))])

    plt.rc("lines", linewidth=1.5)
    plt.stem(x, MLP, label="MLP")
    plt.plot(x, MLP, dashes=[4, 4])
    plt.stem(x, LeNet, linefmt="grey", label="LeNet")
    plt.plot(x,LeNet,color="grey",dashes=[4,4])
    plt.stem(x,VGG,linefmt="pink",label="VGG")
    plt.plot(x,VGG,color="pink",dashes=[4,4])

    plt.xlabel("Epsilon")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.legend()
    plt.savefig("./image1/"+name+".svg")
    plt.show()

def checkout_fig(X,title=None):
    """要求X是一个图片数组且处理成了图片格式,用subplot1x3画出"""
    cnt=0
    for x in X:
        cnt+=1
        plt.subplot(1,3,cnt)
        plt.imshow(x,cmap="gray") 
    plt.show()

def develop_photos(X,epsilon,name=None):
    rowcnt,cnt,num=0,0,len(epsilon)
    for row in X:
        flag=True
        for x in row:
            cnt+=1
            plt.subplot(num,3,cnt)
            plt.xticks([],[])
            plt.yticks([],[])
            
            if flag:
                plt.ylabel(r"$\varepsilon =$"+f"{epsilon[rowcnt]}")
                flag=False

            if rowcnt==0:
                if (cnt-1)%3==0:
                    plt.title("Clean")
                elif (cnt-2)%3==0:
                    plt.title("Perturbed")
                else:
                    plt.title("Noise")
            plt.imshow(x,cmap="gray")
        rowcnt+=1
    plt.suptitle("The Processed Images")
    plt.tight_layout(w_pad=0.2)
    plt.savefig("./image1/"+name+"EmergedPhoto.svg")
    plt.show()    
   
