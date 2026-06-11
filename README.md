# 基于 ResNet18-CBAM 迁移学习的儿童肺炎 X 光图像辅助诊断系统

本项目为《机器学习课程设计》的期末项目源码。本项目基于标准的 PyTorch 深度学习框架，实现了一套从数据集打乱重构、在线复合数据增强、三阶段消融对比实验（自建浅层CNN vs 标准ResNet18全量微调 vs 改进ResNet18-CBAM），到最终独立测试集盲测评估（混淆矩阵、ROC曲线、AUC计算）以及模型决策可解释性分析（Grad-CAM 热力图生成）的完整医学辅助诊断工程。

---

## 1. 运行环境配置 (Requirements)

- **操作系统:** Windows 11
- **Python 版本:** 3.10.2
- **核心依赖库及版本:**
  - `torch` == 2.10.0 (PyTorch 深度学习核心库)
  - `torchvision` == 0.25.0 (用于加载 ResNet 预训练权重与图像变换)
  - `numpy` == 2.2.6 (矩阵计算)
  - `matplotlib` == 3.10.8 (曲线绘制与图像显示)
  - `seaborn` == 0.13.2 (混淆矩阵热力图绘制)
  - `scikit-learn` == 1.7.1 (用于计算分类报告、ROC曲线及AUC值)
  - `opencv-python` == 4.13.0.92 (用于 Grad-CAM 图像叠加处理)
  - `pytorch-grad-cam` == 1.5.5 或更新版本 (用于生成决策热力图)


## 2. 项目目录结构规范
- 项目根目录/  
  - 源代码/  
    - 数据划分.py                 # 8:1:1 数据打乱与重新切分  
    - 数据预处理.py               # 统计划分后数据集并画出分布图  
    - CNN基线模型.py              # 自建 CNN 训练脚本 (10轮)  
    - ResNet18（全量微调）.py     # 标准 ResNet18 微调训练脚本 (10轮)  
    - ResNet-CBAM.py             # ResNet18+CBAM 训练脚本 (10轮)  
    - 模型结果.py                 # 测试集盲测评估 (混淆矩阵、ROC/AUC曲线)  
    - Grad-CAM热力图.py           # 临床决策可视化热力图生成  
  - README.md                    # 本项目说明书  
  - archive/                     # 数据集根目录  
    - chest_xray_split/          # 重新划分后的独立同分布数据集  
      - train/                   # 训练集 (4685张)  
      - val/                     # 验证集 (585张)  
      - test/                    # 测试集 (589张)  


## 3. 运行步骤
      1）数据划分.py
      2）数据预处理.py
      3）CNN基线模型.py
      4）ResNet18（全量微调）.py
      5）ResNet-CBAM.py
      6）模型结果.py
      7）Grad-CAM热力图.py
