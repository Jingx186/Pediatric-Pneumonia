import os
import matplotlib.pyplot as plt
import numpy as np

base_dir = r'D:\机器学习课程设计\期末3\archive\chest_xray_split'

splits = ['train', 'val', 'test']
classes = ['NORMAL', 'PNEUMONIA']

# 1. 动态统计新数据文件夹下的文件数量
data_stats = {'train': {}, 'val': {}, 'test': {}}

for split in splits:
    for cls in classes:
        path = os.path.join(base_dir, split, cls)
        # 只统计图片文件
        num_images = len([f for f in os.listdir(path) if f.endswith(('.jpeg', '.jpg', '.png'))])
        data_stats[split][cls] = num_images

# 2. 打印统计表格数据
print("="*60)
print("             新划分数据集真实分布表 (8:1:1)")
print("="*60)
print(f"数据子集\t正常(NORMAL)\t肺炎(PNEUMONIA)\t合计")
print("-"*60)
for split in splits:
    n_count = data_stats[split]['NORMAL']
    p_count = data_stats[split]['PNEUMONIA']
    total = n_count + p_count
    # 转换为中文名称打印
    split_zh = {'train': '训练集(Train)', 'val': '验证集(Val)', 'test': '测试集(Test)'}[split]
    print(f"{split_zh}\t{n_count}张\t\t{p_count}张\t\t{total}张")
print("="*60)

# 3. 绘制分组柱状图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 准备绘图数据
x_labels = ['训练集 (Train)', '验证集 (Val)', '测试集 (Test)']
normal_values = [data_stats['train']['NORMAL'], data_stats['val']['NORMAL'], data_stats['test']['NORMAL']]
pneumonia_values = [data_stats['train']['PNEUMONIA'], data_stats['val']['PNEUMONIA'], data_stats['test']['PNEUMONIA']]

x = np.arange(len(x_labels)) # 标签位置
width = 0.35 # 柱子宽度

fig, ax = plt.subplots(figsize=(10, 6))
# 画正常和肺炎的两组柱状图
rects1 = ax.bar(x - width/2, normal_values, width, label='正常 (NORMAL)', color='#4CAF50')
rects2 = ax.bar(x + width/2, pneumonia_values, width, label='肺炎 (PNEUMONIA)', color='#F44336')

# 添加标题和标签
ax.set_title('重构数据集样本空间分布图 (8:1:1)', fontsize=16, pad=15)
ax.set_ylabel('图片数量 (张)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=12)
ax.legend(fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.5)

# 在柱状图上方标注具体数值
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3点纵向偏移
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
new_save_path = os.path.join(base_dir, 'new_data_distribution.png')
plt.savefig(new_save_path, dpi=300, bbox_inches='tight')
print(f"\n分组分布图已生成并保存至: {new_save_path}")
plt.show()