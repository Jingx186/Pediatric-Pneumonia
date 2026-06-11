import os
import shutil
import random
from tqdm import tqdm

# 1. 原始数据集路径
src_base = r'D:\机器学习课程设计\期末3\archive\chest_xray'

# 2. 划分后的新数据集保存路径
dest_base = r'D:\机器学习课程设计\期末3\archive\chest_xray_split'

# 定义划分比例 (训练集 80%, 验证集 10%, 测试集 10%)
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

categories = ['NORMAL', 'PNEUMONIA']
splits = ['train', 'val', 'test']

# 创建新文件夹结构
for split in splits:
    for cat in categories:
        os.makedirs(os.path.join(dest_base, split, cat), exist_ok=True)

# 临时存储所有找到的图片
all_images = {'NORMAL': [], 'PNEUMONIA': []}

# 3. 收集原 train, val, test 中的所有图片
print("正在收集并合并所有图片...")
original_splits = ['train', 'val', 'test']
for o_split in original_splits:
    for cat in categories:
        cat_dir = os.path.join(src_base, o_split, cat)
        if os.path.exists(cat_dir):
            for img_name in os.listdir(cat_dir):
                if img_name.endswith(('.jpeg', '.jpg', '.png')):
                    all_images[cat].append(os.path.join(cat_dir, img_name))

# 4. 开始随机打乱并复制到新文件夹
for cat in categories:
    images = all_images[cat]
    random.seed(42)  # 设定随机种子，保证每次运行划分结果一致（可复现性）
    random.shuffle(images)

    total = len(images)
    num_train = int(total * train_ratio)
    num_val = int(total * val_ratio)

    # 划分切片
    train_imgs = images[:num_train]
    val_imgs = images[num_train:num_train + num_val]
    test_imgs = images[num_train + num_val:]

    print(
        f"\n类别 [{cat}] 总数: {total} | 划分 -> 训练集: {len(train_imgs)} | 验证集: {len(val_imgs)} | 测试集: {len(test_imgs)}")


    # 复制文件函数
    def copy_files(file_list, split_name):
        dest_dir = os.path.join(dest_base, split_name, cat)
        for filepath in tqdm(file_list, desc=f"复制 {split_name}/{cat}"):
            shutil.copy(filepath, dest_dir)

    copy_files(train_imgs, 'train')
    copy_files(val_imgs, 'val')
    copy_files(test_imgs, 'test')

print(f"\n数据集重新划分成功！新数据集已保存至: {dest_base}")