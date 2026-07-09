import torch 

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:",device) 

x = torch.rand(3,4).to(device)
print(x,device)