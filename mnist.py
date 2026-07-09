import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader 
from torchvision import datasets, transforms 

BATCH_SIZE = 64 
LEARNING_RATE = 0.001 
EPOCHS = 5 

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))

])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=False,
    transform=transform
)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,shuffle=False)

print(f"训练集样本数：{len(train_dataset)},测试集样本数：{len(test_dataset)}")


class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28,128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
model = MNISTClassifier()
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = LEARNING_RATE )

for epoch in range(EPOCHS):
    print(f"\n---Epoch {epoch+1}/{EPOCHS}")

    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_X, batch_y in train_loader:

        pred = model(batch_X)
        loss = loss_fn(pred, batch_y)

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        total_loss += loss.item()
        _, predicted = torch.max(pred, 1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)


    print(f"Loss:{total_loss/len(train_loader):.4f}) | Acc:{correct/total*100:.2f}%")


model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        pred = model(batch_X)
        _, predicted = torch.max(pred, 1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)

print(f"\n测试集准确率: {correct/total*100:.2f}%")