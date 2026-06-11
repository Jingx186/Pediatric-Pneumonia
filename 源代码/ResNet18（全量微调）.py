import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

# 1. 基础设置与标准数据加载
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用计算设备: {device}。开启全量微调模式，CPU运行预计需要30-60分钟，请耐心等待！")

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

# batch_size 保持 16 减轻内存压力
train_loader = DataLoader(datasets.ImageFolder(train_dir, transform=train_transforms), batch_size=16, shuffle=True)
val_loader = DataLoader(datasets.ImageFolder(val_dir, transform=val_transforms), batch_size=16, shuffle=False)

# 2. 模型构建：全量微调 + Dropout
print("\n正在加载 ResNet18 预训练模型...")
model = models.resnet18(pretrained=True)

# 所有层全部参与训练
for param in model.parameters():
    param.requires_grad = True

num_ftrs = model.fc.in_features
# 在最后一层加入 Dropout(0.5)，强力防止过拟合
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 2)
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
# 学习率缩小十倍 (1e-4)，并加入 L2 正则化 (weight_decay)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
# 学习率调度器。每过 3 个 Epoch，学习率自动减半，帮助模型收敛
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# 3. 开始深度训练
epochs = 10
train_losses, val_losses = [], []
train_accs, val_accs = [], []
best_val_acc = 0.0  # 用于保存最佳模型

print(f"开始全量微调 (Full Fine-Tuning)，总计 {epochs} 轮...")
start_time = time.time()

for epoch in range(epochs):
    epoch_start = time.time()

    # --- 训练 ---
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

    # --- 验证 ---
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

    # 学习率调度器更新
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    epoch_time = time.time() - epoch_start
    print(f"Epoch [{epoch + 1}/{epochs}] 耗时:{epoch_time:.0f}s | "
          f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | 当前LR: {current_lr:.6f}")

    # 只保存验证集准确率最高的那一轮模型(自动早停)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_path = os.path.join(base_dir, 'best_resnet18_full.pth')
        torch.save(model.state_dict(), best_model_path)
        print(f"发现更好模型！已保存至 {best_model_path}")

total_time = (time.time() - start_time) / 60
print(f"\n训练完成! 总耗时: {total_time:.2f} 分钟。最高验证集准确率: {best_val_acc:.4f}")

# 4. 绘制全新的高清对比曲线
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('ResNet18 (全量微调 Full Fine-Tuning) 训练曲线', fontsize=16)

ax1.plot(range(1, epochs + 1), train_losses, label='训练集 Loss', marker='o')
ax1.plot(range(1, epochs + 1), val_losses, label='验证集 Loss', marker='o')
ax1.set_title('损失函数 (Loss) 变化', fontsize=14)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

ax2.plot(range(1, epochs + 1), train_accs, label='训练集 Accuracy', marker='s')
ax2.plot(range(1, epochs + 1), val_accs, label='验证集 Accuracy', marker='s')
ax2.set_title('准确率 (Accuracy) 变化', fontsize=14)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
resnet_curve_save_path = os.path.join(base_dir, 'resnet_full_curves.png')
plt.savefig(resnet_curve_save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"全量微调训练曲线已保存至: {resnet_curve_save_path}")