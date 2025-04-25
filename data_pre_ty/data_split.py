import os 
import cv2  
from tqdm import tqdm
import shutil
import random
img_format = ['.jpg','.png','.PNG','.JPEG','.tif']
def data_split1(imgs_path,labels_path,dataset_root):
    imgs_train = os.path.join(dataset_root,'images','train')
    imgs_val = os.path.join(dataset_root,'images','val')
    labels_train = os.path.join(dataset_root,'labels','train')
    labels_val = os.path.join(dataset_root,'labels','val')
    dirs = [imgs_train,imgs_val,labels_train,labels_val]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
    num1 = 40000
    num2 = 10000
    labels = os.listdir(labels_path)
    random.shuffle(labels)
    train = labels[:num1]
    val = labels[num1:num1+num2]
    for file in tqdm(train):
        txt_file = os.path.join(labels_path,file)
        txt2_file = os.path.join(labels_train,file)
        for format in img_format:
            img_file = os.path.join(imgs_path,os.path.splitext(file)[0] + format)
            img2_file = os.path.join(imgs_train,os.path.splitext(file)[0] + format)
            if os.path.exists(img_file):
                break
        shutil.copy(txt_file,txt2_file)
        shutil.copy(img_file,img2_file)

    for file in tqdm(val):
        txt_file = os.path.join(labels_path,file)
        txt2_file = os.path.join(labels_val,file)
        for format in img_format:
            img_file = os.path.join(imgs_path,os.path.splitext(file)[0] + format)
            img2_file = os.path.join(imgs_val,os.path.splitext(file)[0] + format)
            if os.path.exists(img_file):
                break
        shutil.copy(txt_file,txt2_file)
        shutil.copy(img_file,img2_file)

def data_split2(imgs_dir,train_file,val_file):
    # 设置随机种子确保结果可复现
    random.seed(42)

    # 数据集中所有图像的路径
    images_paths = [os.path.join(imgs_dir, image) for image in os.listdir(imgs_dir) if image.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 打乱路径顺序
    random.shuffle(images_paths)

    # 按照9:1的比例划分训练集和验证集
    split_point = int(0.9 * len(images_paths))
    train_images = images_paths[:split_point]
    val_images = images_paths[split_point:]

    with open(train_file, 'w') as f:
        for path in train_images:
            f.write(f"{path}\n")

    with open(val_file, 'w') as f:
        for path in val_images:
            f.write(f"{path}\n")

    print(f"训练集样本数量: {len(train_images)}, 验证集样本数量: {len(val_images)}")


if __name__ == '__main__':
    # dataset_root = r'/mnt/pai-storage-ceph-hdd/Dataset/Vehicle/QIDIAN/color_detect2'
    # imgs_path = r'/mnt/pai-storage-ceph-hdd/Dataset/Vehicle/QIDIAN/images_all'
    # labels_path = r'/mnt/pai-storage-ceph-hdd/Dataset/Vehicle/QIDIAN/color_detect2/labels_all'
    # data_split1(imgs_path,labels_path,dataset_root)
    for key in range(6):
        images_dir = fr"/mnt/pai-storage-ceph-hdd/Dataset/Vehicle/QIDIAN/color_class/images_cls2/{key}"
        train_file = fr"/mnt/pai-storage-ceph-hdd/Dataset/Vehicle/QIDIAN/color_class/dataset/pos_cls416_all/train.txt"
        val_file = fr"/mnt/pai-storage-ceph-hdd/Dataset/Vehicle/QIDIAN/color_class/dataset/pos_cls416_all/val.txt"
        data_split2(images_dir,train_file,val_file)
        print("All images have been moved successfully.")
