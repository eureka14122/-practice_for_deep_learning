import torch 

import torch.nn as nn 

from torch.utils.data import TensorDataset, DataLoader


device = "cuda" if torch.cuda.is_available() else "cpu"

print("device:",device )

class TinyNet(nn.Module):

    def __init__(self): 
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4,16),
            nn.ReLU(),
            nn.Linear(16,2),
        )
    def forward(self,x):
        return self.net(x)
        
model = TinyNet().to(device)


X = torch.randn(100, 4)
y = (X[:, 0] + X[:, 1] > 0).long()

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


for xb, yb in loader:
    xb = xb.to(device)
    yb = yb.to(device)
    print(xb.shape, yb.shape)
    break

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for xb, yb in loader:
    xb = xb.to(device)
    yb = yb.to(device)
    pred = model(xb) # 1. 前向传播
    loss = loss_fn(pred, yb) # 2. 计算损失
    optimizer.zero_grad() # 3. 清空旧梯度
    loss.backward() # 4. 反向传播
    optimizer.step() # 5. 更新参数