import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# 0. 定义CBAM 注意力结构
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

class ResNet18_CBAM(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18_CBAM, self).__init__()
        resnet = models.resnet18(pretrained=False)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.cbam = CBAM(in_planes=512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# 1. 基础设置与数据加载
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_dir = r'D:\机器学习课程设计\期末3\archive\chest_xray_split'
test_dir = os.path.join(base_dir, 'test')  # 使用独立测试集进行盲测

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 2. 加载最强模型权重
print("正在加载保存的最优 ResNet18-CBAM 模型权重...")
model = ResNet18_CBAM(num_classes=2)
model_path = os.path.join(base_dir, 'best_resnet18_cbam.pth')
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# 3. 进行推理预测
print("正在对全新测试集进行诊断评估，请稍候...")
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)

        _, preds = torch.max(outputs, 1)
        probs = F.softmax(outputs, dim=1)
        pos_probs = probs[:, 1]  # 提取预测为肺炎的概率

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(pos_probs.cpu().numpy())

# 4. 生成报告与混淆矩阵
print("\n" + "=" * 50)
print("             终极分类评估报告 (Classification Report)")
print("=" * 50)
print(classification_report(all_labels, all_preds, target_names=['正常(NORMAL)', '肺炎(PNEUMONIA)']))

cm = confusion_matrix(all_labels, all_preds)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['预测: 正常', '预测: 肺炎'],
            yticklabels=['真实: 正常', '真实: 肺炎'],
            annot_kws={"size": 16})

plt.title('ResNet18-CBAM 创新模型混淆矩阵', fontsize=18, pad=20)
plt.yticks(rotation=0, fontsize=12)
plt.xticks(fontsize=12)

cm_save_path = os.path.join(base_dir, 'confusion_matrix.png')
plt.savefig(cm_save_path, dpi=300, bbox_inches='tight')
print(f"\n混淆矩阵图已保存至: {cm_save_path}")
plt.show()

# 5. 绘制 ROC 曲线
print("\n正在生成 ROC 曲线...")
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='red', lw=2, label=f'ResNet18-CBAM ROC 曲线 (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率 (False Positive Rate)', fontsize=14)
plt.ylabel('真阳性率 (True Positive Rate)', fontsize=14)
plt.title('肺炎分类 受试者工作特征曲线 (ROC)', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

roc_save_path = os.path.join(base_dir, 'roc_curve.png')
plt.savefig(roc_save_path, dpi=300, bbox_inches='tight')
print(f"ROC曲线图已保存至: {roc_save_path}")
plt.show()