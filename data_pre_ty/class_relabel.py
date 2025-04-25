import shutil
import json 
import os 
from tqdm import tqdm
import cv2
with open('/home/tianyuan/workspace/sqyolov8/data_pre1/merged_20240402.json', 'r') as file:
    data1 = json.load(file)
with open('/home/tianyuan/workspace/sqyolov8/data_pre1/Mar_detect.json', 'r') as file:
    data2 = json.load(file)
dim = (128,128)
for da1 in tqdm(data1):
    path1 = da1['img_path']
    _, name = os.path.split(path1)
    for da2 in data2:
        # if name == da2['img_name'] and da1['chcl'] == '2':
        #     dir2 = r'/mnt/pai-storage-ceph-hdd/Dataset/Vehicle/QIDIAN/Mar_class_relabel'
        #     path2 = os.path.join(dir2,name)
        #     img = cv2.imread(da2['img_path'])
        #     box_info= da2['box_info']
        #     if box_info:
        #         box_data = box_info[0]['box']
        #         crop_image = img[box_data[1]:box_data[3], box_data[0]:box_data[2]]
        #         cv2.imwrite(path2,crop_image)
        if name == da2['img_name'] and da1['chcl'] == '1':
            dir1 = r'/mnt/pai-storage-ceph-hdd/Dataset/Vehicle/QIDIAN/color_class_Mar/images_crop'
            path1 = os.path.join(dir1,name)
            img = cv2.imread(da2['img_path'])
            box_info= da2['box_info']
            if box_info:
                box_data = box_info[0]['box']
                crop_image = img[box_data[1]:box_data[3], box_data[0]:box_data[2]]
                resized_image = cv2.resize(crop_image, dim, interpolation=cv2.INTER_AREA)
                cv2.imwrite(path1,resized_image)
        

