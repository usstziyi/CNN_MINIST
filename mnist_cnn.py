import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# 检查MPS设备是否可用
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第一层卷积: 1个输入通道, 32个输出通道,  kernel_size=3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # 第二层卷积: 32个输入通道, 64个输出通道, kernel_size=3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 池化层: kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层: 7*7*64 -> 128
        self.fc1 = nn.Linear(7*7*64, 128)
        # 全连接层: 128 -> 10 (10个类别)
        self.fc2 = nn.Linear(128, 10)
        # Dropout层
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 第一层卷积 + ReLU + 池化
        x = self.pool(F.relu(self.conv1(x)))
        # 第二层卷积 + ReLU + 池化
        x = self.pool(F.relu(self.conv2(x)))
        # 展平
        x = x.view(-1, 7*7*64)
        # 全连接层 + ReLU + Dropout
        x = self.dropout(F.relu(self.fc1(x)))
        # 输出层
        x = self.fc2(x)
        return x

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)

test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# 创建数据加载器
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型、损失函数和优化器
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 100 == 99:
            print(f'[{epoch}, {batch_idx+1}] loss: {running_loss/100:.3f}')
            running_loss = 0.0

# 测试模型
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')

# 开始训练和测试
epochs = 5
start_time = time.time()

for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, criterion, epoch)
    test(model, device, test_loader, criterion)

end_time = time.time()
print(f'训练和测试总用时: {end_time - start_time:.2f}秒')

# 保存模型
torch.save(model.state_dict(), 'mnist_cnn_model.pth')
print('模型已保存为 mnist_cnn_model.pth')