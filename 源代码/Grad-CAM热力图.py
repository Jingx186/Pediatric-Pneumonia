import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# 导入 Grad-CAM 相关库
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# 0. 定义 CBAM 注意力结构
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

# 1. 基础设置与模型加载
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_dir = r'D:\机器学习课程设计\期末3\archive\chest_xray_split'

# 加载模型
print("正在加载最优 ResNet18-CBAM 模型并准备生成热力图...")
model = ResNet18_CBAM(num_classes=2)
model_path = os.path.join(base_dir, 'best_resnet18_cbam.pth')
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# 2. 准备几张要分析的图片
pneumonia_test_dir = os.path.join(base_dir, 'test', 'PNEUMONIA')
img_names = [f for f in os.listdir(pneumonia_test_dir) if f.endswith('.jpeg')][:3]  # 取前3张

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

# 3. 配置 Grad-CAM
target_layers = [model.features[-1][-1]]
cam = GradCAM(model=model, target_layers=target_layers)
targets = [ClassifierOutputTarget(1)]

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('ResNet18-CBAM 肺炎辅助诊断可解释性分析 (Grad-CAM热力图)', fontsize=18)

print("正在生成可解释性热力图...")

for i, img_name in enumerate(img_names):
    img_path = os.path.join(pneumonia_test_dir, img_name)

    # 1. 读取原图
    raw_pil_img = Image.open(img_path).convert('RGB').resize((224, 224))
    rgb_img = np.float32(raw_pil_img) / 255

    # 2. 预处理
    input_tensor = transform(raw_pil_img).unsqueeze(0).to(device)

    # 3. 生成掩码
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    # 4. 叠加显示
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # 画原图
    axes[0, i].imshow(rgb_img)
    axes[0, i].set_title(f"原始X光片 - 样本 {i + 1}", fontsize=14)
    axes[0, i].axis('off')

    # 画热力图
    axes[1, i].imshow(visualization)
    axes[1, i].set_title(f"CBAM 注意力聚焦区域 - 样本 {i + 1}", fontsize=14)
    axes[1, i].axis('off')

plt.tight_layout()
cam_save_path = os.path.join(base_dir, 'grad_cam_visualization.png')
plt.savefig(cam_save_path, dpi=300, bbox_inches='tight')
print(f"热力图已保存至: {cam_save_path}")
plt.show()