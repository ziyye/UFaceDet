import shutil
import cv2
import numpy as np
import itertools
import glob
import os 
import json

from tqdm import tqdm

chars = 'rbgy'
# 使用itertools.permutations生成所有排列，参数2表示排列的长度
permutations = itertools.permutations(chars, len(chars))
# 转换每个排列元组为字符串，并放入到一个集合中，以去除可能的重复项
unique_combinations = {''.join(p) for p in permutations}
# 打印结果
color_dir = {}
for combo in sorted(unique_combinations):
    color_dir[combo] = []
# print(color_dir)
# 打印组合总数
# print(f"Total combinations: {len(unique_combinations)}")

json_path = r'/mnt/pai-storage-ceph-hdd/Dataset/Vehicle/QIDIAN/color_class/images_crop_0523/crops/color.json'
# json_path = r'/mnt/pai-storage-1/tianyuan/workspace/qidian/sqyolov8/test_imgs/color.json'

img_dir_list = [
    f'/mnt/pai-storage-ceph-hdd/Dataset/Vehicle/QIDIAN/color_class/images_crop_0523/crops/color/*',
]
# img_dir_list = [
    # f'/mnt/pai-storage-1/tianyuan/workspace/qidian/sqyolov8/test_imgs/cls/*',
# ]

imgLists = []
for img_dir in img_dir_list:
    imgLists += glob.glob(img_dir)
imgLists = list(filter(lambda path: os.path.splitext(path)[-1].lower() in ['.jpg', '.jpeg', '.png'], imgLists))

# print(len(imgLists))

# copy_path = r'/mnt/pai-storage-1/tianyuan/workspace/qidian/sqyolov8/test_imgs/cls'
 # a = os.path.split(image_path)
    # shutil.copy(image_path,os.path.join(copy_path,os.path.split(image_path)[-1]))
for image_path in tqdm(imgLists):
   
    image = cv2.imread(image_path)
    # 转换到HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义颜色的HSV阈值范围
    # 注意: 这些值需要根据你的图片特定情况调整
    color_ranges = {
        'red': ((0, 50, 50), (10, 255, 255)),
        'yellow': ((20, 100, 100), (30, 255, 255)),
        'green': ((40, 50, 50), (80, 255, 255)),
        'blue': ((100, 50, 50), (140, 255, 255))
    }

    # 初始化颜色条纹中心y坐标字典
    color_strip_centers = {}

    for color, (lower, upper) in color_ranges.items():
        # 创建阈值掩模
        mask = cv2.inRange(image_hsv, lower, upper)
        
        # 应用掩模获得每种颜色的区域
        color_region = cv2.bitwise_and(image, image, mask=mask)
        
        # 找出每种颜色区域的质心
        M = cv2.moments(mask)
        if M["m00"] != 0:  # 防止除以0
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # 保存y坐标
            color_strip_centers[color] = cY
        else:
            color_strip_centers[color] = None
    # 根据y坐标排序颜色
    sorted_colors = sorted(color_strip_centers.items(), key=lambda x: x[1]) if None not in color_strip_centers.values() else ''
    color = ''.join([color[0] for color, _ in sorted_colors])
    if color in color_dir.keys():
        color_dir[color].append(image_path)
    # 打印顺序
with open(json_path, 'w', encoding='utf-8') as fw:
    json.dump(color_dir, fw, ensure_ascii=False, indent=4)