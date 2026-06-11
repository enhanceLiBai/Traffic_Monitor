import os
import random
import shutil

# 1. 定义你的原始图片目录和想要输出的分组目录
src_dir = "/root/autodl-tmp/FlowPlate-Analyzer/CCPD2019and2020/ccpd_base"  # ⚠️请把这里改成你解压后的实际路径
output_base = "/root/autodl-tmp/FlowPlate-Analyzer/data"

train_dir = os.path.join(output_base, "train")
val_dir = os.path.join(output_base, "val")
test_dir = os.path.join(output_base, "test")

# 创建目标文件夹
for d in [train_dir, val_dir, test_dir]:
    os.makedirs(d, exist_ok=True)

# 2. 获取所有 jpg 图片并打乱
all_images = [f for f in os.listdir(src_dir) if f.endswith('.jpg')]
random.seed(42)  # 固定随机种子，确保每次划分结果一致
random.shuffle(all_images)

total_count = len(all_images)
print(f"数据集总计找到 {total_count} 张图片。开始计算划分数量...")

# 3. 按 8:1:1 计算各个集合的边界索引
train_end = int(total_count * 0.8)
val_end = train_end + int(total_count * 0.1)

train_images = all_images[:train_end]
val_images = all_images[train_end:val_end]
test_images = all_images[val_end:]

print(f"计算完毕！训练集: {len(train_images)}张, 验证集: {len(val_images)}张, 测试集: {len(test_images)}张")

# 4. 开始移动（或复制）文件
def move_files(files, dest_folder):
    for f in files:
        # 如果数据量极大，建议用 shutil.move（剪切），省空间；
        # 如果想留备份，用 shutil.copy（复制）
        shutil.move(os.path.join(src_dir, f), os.path.join(dest_folder, f))

print("正在将图片分发到各个标准文件夹，请稍候...")
move_files(train_images, train_dir)
move_files(val_images, val_dir)
move_files(test_images, test_dir)
print("🎉 数据集分组成功！")