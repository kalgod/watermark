import os
import random
import shutil

# 设置随机种子以确保每次运行结果一致
random.seed(42)

# 输入文件夹和输出文件夹路径
input_folder = './val2014'
output_train_folder = './coco/train/train_class'
output_val_folder = './coco/val/val_class'

# 确保train和val文件夹存在
os.makedirs(output_train_folder, exist_ok=True)
os.makedirs(output_val_folder, exist_ok=True)

# 获取test2017文件夹下的所有文件
file_list = os.listdir(input_folder)

# 打乱文件列表顺序
random.shuffle(file_list)

# 从打乱后的文件列表中选择一部分作为train文件
train_files = file_list[:10000]

# 从剩下的文件中选择一部分作为val文件
val_files = file_list[10000:11000]

# 复制选中的train文件到output_train_folder
for file in train_files:
    shutil.copy(os.path.join(input_folder, file), output_train_folder)

# 复制选中的val文件到output_val_folder
for file in val_files:
    shutil.copy(os.path.join(input_folder, file), output_val_folder)
