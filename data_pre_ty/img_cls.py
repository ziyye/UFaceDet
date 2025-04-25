import shutil
import json 
import os 
from tqdm import tqdm
with open('/mnt/pai-storage-1/tianyuan/workspace/qidian/sqyolov8/data_pre1/label_20240402.json', 'r') as file:
    data1 = json.load(file)

imgs_dir = r'/mnt/pai-storage-ceph-hdd/Dataset/Vehicle/QIDIAN/color_class_Mar/images_crop'
imgs_cls_dirs = r'/mnt/pai-storage-ceph-hdd/Dataset/Vehicle/QIDIAN/color_class_Mar/images_cls'
for img in tqdm(os.listdir(imgs_dir)):
    img1_path = os.path.join(imgs_dir,img)
    for da in data1:
        if img == da['img_name']:
            dir1 = "".join(da['color'])
            cls_dir = os.path.join(imgs_cls_dirs,dir1)
            if not os.path.exists(cls_dir):
                os.mkdir(cls_dir)
            img2_path = os.path.join(cls_dir,img)
            shutil.copy(img1_path,img2_path)
