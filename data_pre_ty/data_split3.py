import os
import shutil
import random
from tqdm import tqdm

def split_dataset(data_root, train_ratio=0.9):
    images_path = os.path.join(data_root, 'images')
    labels_path = os.path.join(data_root, 'labels')
    
    train_images_path = os.path.join(data_root, 'train', 'images')
    train_labels_path = os.path.join(data_root, 'train', 'labels')
    val_images_path = os.path.join(data_root, 'val', 'images')
    val_labels_path = os.path.join(data_root, 'val', 'labels')
    
    # 创建目标文件夹
    for path in [train_images_path, train_labels_path, val_images_path, val_labels_path]:
        os.makedirs(path, exist_ok=True)
    
    # 获取所有标签文件
    labels = [f for f in os.listdir(labels_path) if f.endswith('.txt')]
    random.shuffle(labels)
    
    # 计算分割点
    split_point = int(len(labels) * train_ratio)
    train_labels = labels[:split_point]
    val_labels = labels[split_point:]
    
    # 复制训练集
    print("正在复制训练集文件...")
    for label_file in tqdm(train_labels, desc="训练集进度"):
        shutil.copy(os.path.join(labels_path, label_file), os.path.join(train_labels_path, label_file))
        image_file = os.path.splitext(label_file)[0] + '.jpg'  # 假设图像格式为.jpg
        shutil.copy(os.path.join(images_path, image_file), os.path.join(train_images_path, image_file))
    
    # 复制验证集
    print("正在复制验证集文件...")
    for label_file in tqdm(val_labels, desc="验证集进度"):
        shutil.copy(os.path.join(labels_path, label_file), os.path.join(val_labels_path, label_file))
        image_file = os.path.splitext(label_file)[0] + '.jpg'  # 假设图像格式为.jpg
        shutil.copy(os.path.join(images_path, image_file), os.path.join(val_images_path, image_file))
    
    print(f"训练集样本数量: {len(train_labels)}, 验证集样本数量: {len(val_labels)}")

if __name__ == '__main__':
    data_root = '/mnt/pai-storage-8/tianyuan/smoke/data/dataset/open_data_1/train'  # 替换为你的数据主文件夹路径
    split_dataset(data_root)