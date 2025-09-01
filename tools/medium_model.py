
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class SmallMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 250),
        nn.ReLU(),
        nn.Linear(250, 200),
        nn.ReLU(),
        nn.Linear(200, 10)  # 10 logits # 10 logits #10 logits 
)


    def forward(self, x):
        return self.net(x)


transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

model = SmallMNIST().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),weight_decay=1e-3, lr=1e-3)
# add a regularization pararameter 
#  from 1e 
# adapt

def train(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    correct = 0
    for x, y in tqdm(loader, desc=f"Epoch {epoch}"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
    acc = correct / len(loader.dataset)
    print(f"Train accuracy: {acc:.4f} | Loss: {total_loss/len(loader):.4f}")


def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
    acc = correct / len(loader.dataset)
    print(f"Test accuracy: {acc:.4f}")
    return acc


for epoch in range(10):  
    train(model, train_loader, optimizer, criterion, epoch)
    evaluate(model, test_loader)

torch.save(model.state_dict(), "medium250_200_size_net.pth")
print("Model saved as 'medium250_200_size_net.pth'")


import torch.onnx


model = SmallMNIST()
model.load_state_dict(torch.load("medium250_200_size_net.pth", map_location=torch.device('cpu')))
model.eval()


dummy_input = torch.randn(1, 1, 28, 28)

# Export to ONNX
torch.onnx.export(
    model, 
    dummy_input, 
    "medium250_200_size_net.onnx",
    input_names=['input'], 
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}},
    opset_version=11
)

print("Exported to 'medium250_200_size_net.onnx'")

