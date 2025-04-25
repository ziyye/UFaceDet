import json
import shutil
import os
from tqdm import tqdm
# 示例JSON数据
with open('/mnt/pai-storage-ceph-hdd/Dataset/Vehicle/QIDIAN/color_class/images_cls2/imgs_cls2_0422_4.json', 'r') as file:
    data = json.load(file)
num  = 0 
base_path = "/mnt/pai-storage-ceph-hdd/Dataset/Vehicle/QIDIAN/color_class/images_cls2"
for item in tqdm(data):
    # 获取位置分类 'pos_cls'
    pos_cls = item['pos_cls']

    # 为此位置分类构建特定的文件夹路径
    dest_path = os.path.join(base_path, pos_cls)

    # 确保文件夹存在
    if not os.path.exists(dest_path):
        os.makedirs(dest_path, exist_ok=True)

    # 获取图片的原始路径
    src_img_path = item['img_path']

    # 构建目标图片路径
    dest_img_path = os.path.join(dest_path, item['img_name'])

    # 复制图片到目标文件夹
    # if pos_cls == '4':
        # num += 1
    shutil.copy(src_img_path, dest_img_path)
# print(num)
    # print(f"图片 {item['img_name']} 已经被复制到文件夹 {pos_cls}")