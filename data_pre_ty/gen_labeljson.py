import os
import shutil
from pathlib import Path

# 定义路径
A_labels_dir = '/mnt/pai-storage-8/tianyuan/facedet/sqyolov8/runs/predict_pt/qrcode2/labels'  # 人脸标注文件夹
B_images_dir = '/data01/data/qrcode/synthetise/val/images'  # 目标图片文件夹
B_labels_dir = '/data01/data/qrcode/synthetise/val/labels'  # 目标标注文件夹
C_images_dir = '/mnt/pai-storage-7/cephssd/jieshen/data/qrcode/synthetise/images/val'  # 源图片文件夹
C_labels_dir = '/mnt/pai-storage-7/cephssd/jieshen/data/qrcode/synthetise/labels/val'  # 源标注文件夹

# 确保目标文件夹存在
os.makedirs(B_images_dir, exist_ok=True)
os.makedirs(B_labels_dir, exist_ok=True)

# 遍历A文件夹中的所有txt文件
for txt_file in Path(A_labels_dir).glob('*.txt'):
    # 获取文件名（不含扩展名）
    file_stem = txt_file.stem
    
    # 构建源文件和目标文件路径
    source_image = os.path.join(C_images_dir, f'{file_stem}.jpg')
    source_label = os.path.join(C_labels_dir, f'{file_stem}.txt')
    target_image = os.path.join(B_images_dir, f'{file_stem}.jpg')
    target_label = os.path.join(B_labels_dir, f'{file_stem}.txt')
    
    # 检查源文件是否存在
    if os.path.exists(source_image) and os.path.exists(source_label):
        # 复制文件
        shutil.copy2(source_image, target_image)
        shutil.copy2(source_label, target_label)
        # print(f'已复制: {file_stem}')
    else:
        print(f'警告: {file_stem} 的源文件不存在')

print('处理完成！')