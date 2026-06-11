import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

# 0. 加入模块：CBAM 注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

# 1. 定义融合模型：ResNet18 + CBAM
class ResNet18_CBAM(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18_CBAM, self).__init__()
        # 加载 ResNet18
        resnet = models.resnet18(pretrained=True)
        # 剥离最后两层
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        # ResNet18 输出特征图的通道数是 512
        self.cbam = CBAM(in_planes=512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # 强力防过拟合
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)  # 施加注意力
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 2. 基础设置与数据加载
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用设备: {device}。开启 ResNet18+CBAM 创新训练！")

# 1. 路径
base_dir = r'D:\机器学习课程设计\期末3\archive\chest_xray_split'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

train_loader = DataLoader(datasets.ImageFolder(train_dir, transform=train_transforms), batch_size=16, shuffle=True)
val_loader = DataLoader(datasets.ImageFolder(val_dir, transform=val_transforms), batch_size=16, shuffle=False)

# 3. 初始化与优化器配置
model = ResNet18_CBAM(num_classes=2).to(device)

# 全量微调，释放所有参数
for param in model.parameters():
    param.requires_grad = True

# 放弃加权交叉熵，使用正常的，因为要追求综合的高准确率
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# 4. 开始训练
epochs = 10
train_losses, val_losses = [], []
train_accs, val_accs = [], []
best_val_acc = 0.0

start_time = time.time()
for epoch in range(epochs):
    epoch_start = time.time()

    # 训练
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    # 验证
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
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

    scheduler.step()

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch [{epoch + 1}/{epochs}] 耗时:{time.time() - epoch_start:.0f}s | "
          f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(base_dir, 'best_resnet18_cbam.pth'))
        print(f"   最佳模型保存！准确率: {best_val_acc:.4f}")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('ResNet18+CBAM 网络训练曲线', fontsize=16)
ax1.plot(range(1, epochs + 1), train_losses, label='训练集 Loss', marker='o')
ax1.plot(range(1, epochs + 1), val_losses, label='验证集 Loss', marker='o')
ax1.set_title('损失函数变化')
ax1.legend()
ax2.plot(range(1, epochs + 1), train_accs, label='训练集 Accuracy', marker='s')
ax2.plot(range(1, epochs + 1), val_accs, label='验证集 Accuracy', marker='s')
ax2.set_title('准确率变化')
ax2.legend()
plt.savefig(os.path.join(base_dir, 'resnet_cbam_curves.png'), dpi=300, bbox_inches='tight')
plt.show()