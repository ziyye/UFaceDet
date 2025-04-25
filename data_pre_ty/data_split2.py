import os
import shutil
from tqdm import tqdm
import random
from PIL import Image
import glob
import os

color_dic = {0: 'bgry',1:'bgyr',2: 'bryg',3: 'byrg',4: 'gbry',5: 'gbyr',
            6:'gryb',7: 'gybr',8: 'gyrb',9: 'rbgy',10: 'rbyg',11: 'rgby',
            12: 'rybg',13: 'rygb',14:'ybgr',15:'ygbr',16:'yrbg',17:'yrgb'}

for name in color_dic.values():

    # 创建文件夹
    train_dir = f'/mnt/pai-storage-ceph-hdd/Dataset/Vehicle/QIDIAN/color_class/dataset/color_cls18/train/{name}'
    val_dir = f'/mnt/pai-storage-ceph-hdd/Dataset/Vehicle/QIDIAN/color_class/dataset/color_cls18/val/{name}'
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)  
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    img_dir_list = [
            f'/mnt/pai-storage-ceph-hdd/Dataset/Vehicle/QIDIAN/color_class/images_cls/{name}/*',
        ]
    imgLists = []
    for img_dir in img_dir_list:
        imgLists += glob.glob(img_dir)
    imgLists = list(filter(lambda path: os.path.splitext(path)[-1].lower() in ['.jpg', '.jpeg', '.png'], imgLists))
    random.shuffle(imgLists)
    train = imgLists[:2000]
    val = imgLists[-500:]
    for image_path in tqdm(train):
        new_path = os.path.join(train_dir,os.path.split(image_path)[-1])
        shutil.copy(image_path,new_path)
    for image_path in tqdm(val):
        new_path = os.path.join(val_dir,os.path.split(image_path)[-1])
        shutil.copy(image_path,new_path)
