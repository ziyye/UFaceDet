import shutil
import json
import os 
from tqdm import tqdm
# import cv2
with open('/mnt/pai-storage-1/tianyuan/workspace/qidian/sqyolov8/data_pre_ty/relabel_2.json', 'r') as file:
    data = json.load(file)

img_path = r'/mnt/pai-storage-ceph-hdd/Dataset/Vehicle/QIDIAN/color_class_Mar/images_cls'
# 模拟输入数据
for da in tqdm(data):
    point_info = da['point_info']
    img_path1 = da['img_path']
    img_name = da['img_name']
    # 根据point的y值排序
    sorted_point_info = sorted(point_info, key=lambda item: item['point']['y'])
    # 提取排序后的point_value
    sorted_point_values = [item['point_value'][0] for item in sorted_point_info]
    dir2 = ''.join(sorted_point_values)
    dir_path2 = os.path.join(img_path,dir2)
    if not os.path.exists(dir_path2):
         os.mkdir(dir_path2)
    img_path2 = os.path.join(img_path,dir2,img_name)
    if os.path.isfile(img_path1):
        # 移动文件到目标目录
        shutil.move(img_path1, img_path2)
        # print(f"Moving {filename} to {target_dir}")
    else:
        print(f"Skipping {img_name}, it's not a file.")
    # 打印结果
    print('ok')
