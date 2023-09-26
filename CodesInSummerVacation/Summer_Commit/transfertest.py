import Models as md
import fetch_data as fd
import attacker as Adv
import torch
import EYE as eye

device=torch.device('cuda:0')
net=md.MLP(dropout=0.15,load_dict=True).to(device)
epsilon=[0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
test_iter=fd.load_data_fashion_mnist_test()

result1,result2=Adv.attack(net,test_iter,epsilon,model_name="MLP")
eye.transfer_test(epsilon,result1,title="Validate Transferability of Adversaries",name="Validate_Trans_MLP",ylabel="Adversary Attack Accuracy")
eye.transfer_test(epsilon,result2,title="Comparison with Noise",name="Noisy_Validate_MLP",ylabel="Added Noise Accuracy")