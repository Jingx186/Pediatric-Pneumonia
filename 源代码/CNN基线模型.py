import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

# 1. 基础设置与数据加载
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用计算设备: {device}")

base_dir = r'D:\机器学习课程设计\期末3\archive\chest_xray_split'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

# 使用ImageNet三通道标准归一化参数
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # 统一为 224
    transforms.RandomRotation(15),  # 数据增强：旋转
    transforms.RandomHorizontalFlip(),  # 数据增强：翻转
    transforms.ToTensor(),  # 转化为张量：并归一化到 [0,1]
    transforms.Normalize(imagenet_mean, imagenet_std)
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # 统一为 224
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

# 使用 ImageFolder 直接从文件夹结构读取数据
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

# DataLoader：批量喂给模型数据，每次喂 32 张
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"训练集图像总数: {len(train_dataset)}")
print(f"验证集图像总数: {len(val_dataset)}")

# 2. 定义基础版卷积神经网络 (Simple CNN)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 提取特征的卷积层
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # 分类器 (全连接层)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  # 防止过拟合
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()  # 损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 优化器

# 3. 开始训练模型
epochs = 10
train_losses, val_losses = [], []
train_accs, val_accs = [], []

print("\n开始训练 Baseline CNN...")
start_time = time.time()

for epoch in range(epochs):
    epoch_start = time.time()

    # --- 训练阶段 ---
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # 清空梯度
        outputs = model(images)  # 预测
        loss = criterion(outputs, labels)  # 计算误差
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    # --- 验证阶段 ---
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():  # 验证时不计算梯度
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total

    # 记录数据画图用
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    epoch_time = time.time() - epoch_start
    print(f"Epoch [{epoch + 1}/{epochs}] 耗时:{epoch_time:.0f}s | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

total_time = (time.time() - start_time) / 60
print(f"\n训练完成! 总耗时: {total_time:.2f} 分钟")

# 4. 绘制训练曲线并保存
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Baseline CNN 训练与验证曲线', fontsize=16)

# 画 Loss 曲线
ax1.plot(range(1, epochs + 1), train_losses, label='训练集 Loss', marker='o')
ax1.plot(range(1, epochs + 1), val_losses, label='验证集 Loss', marker='o')
ax1.set_title('损失函数 (Loss) 变化', fontsize=14)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

# 画 Accuracy 曲线
ax2.plot(range(1, epochs + 1), train_accs, label='训练集 Accuracy', marker='s')
ax2.plot(range(1, epochs + 1), val_accs, label='验证集 Accuracy', marker='s')
ax2.set_title('准确率 (Accuracy) 变化', fontsize=14)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
curve_save_path = os.path.join(base_dir, 'baseline_curves.png')
plt.savefig(curve_save_path, dpi=300, bbox_inches='tight')
print(f"训练曲线已保存至: {curve_save_path}")
plt.show()