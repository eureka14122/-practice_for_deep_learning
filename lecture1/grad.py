import torch 

w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)
y = w * x + 1 

y.backward()
print(w.grad) 