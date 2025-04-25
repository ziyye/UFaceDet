import glob
import json
import random
import shutil
import os

from tqdm import tqdm
def count_is_need(json_file):
    # 初始化计数器
    count0 = 0
    count1 = 0
    imgList = []
    # 打开并加载JSON文件
    with open(json_file, 'r',encoding='utf-8') as file:
        data = json.load(file)
         
        # 遍历列表中的所有项
        for item in data:
            # 检查"is_need"字段是否为"1"
            # if item.get("is_need") == "0":
            #     count0 += 1
            if item.get("cls_label") == "1":
                # count1 += 1 
                imgList.append(item["img_path"])
    return imgList

# JSON文件路径
file_path = r'/mnt/pai-storage-1/tianyuan/workspace/smoke/sqyolov8/select/json_files/huiliu_phone_0716.json'

# 调用函数并打印结果
imgList = count_is_need(file_path)
print(len(imgList))

# img_dir_list = [
#             f'/mnt/pai-storage-ceph-hdd/Dataset/Vehicle/QIDIAN/pos_class/dataset/pos_cls416_3500/train/4/*',
#         ]
# imgList = []
# for img_dir in img_dir_list:
#     imgList += glob.glob(img_dir)
# imgList = list(filter(lambda path: os.path.splitext(path)[-1].lower() in ['.jpg', '.jpeg', '.png'], imgList))

# random.shuffle(imgList)
imgdir1 =  r'/mnt/pai-storage-ceph-hdd/Dataset/smoke_ty/data_huiliu/202312after/head_row/2'
imgdir2 =  r'/mnt/pai-storage-ceph-hdd/Dataset/smoke_ty/data_huiliu/202312after/head_row/2_phone'
if not os.path.exists(imgdir2):
    os.makedirs(imgdir2)
for img_path in tqdm(imgList):
    img_path1 = os.path.join(imgdir1,os.path.split(img_path)[-1])
    img_path2 = os.path.join(imgdir2,os.path.split(img_path)[-1])
    if os.path.exists(img_path1):
        # os.remove(img_path)
        shutil.move(img_path1,img_path2)