import torch 
device = "cuda" if torch.cuda.is_available() else "cpu"



x = torch.ones(4, 4)
s = x.sum()

print(s)

