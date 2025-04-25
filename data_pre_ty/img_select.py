import json
import shutil
import os
import glob
from tqdm import tqdm
import random
import cv2
import numpy as np
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def img_select(n_class=None,k = 100):
    for i in range(n_class):
        if i == k:
            continue
        txt_path = fr'/mnt/pai-storage-1/tianyuan/workspace/smoke/sqyolov8/select/0806_{i}.txt'

        # 目标文件夹的路径
        # source_path = fr'/mnt/pai-storage-ceph-hdd/Dataset/smoke_ty/data_huiliu/202312after'
        # select_path = fr'/mnt/pai-storage-ceph-hdd/Dataset/smoke_ty/data_huiliu/202312after/{i}'
        select_path = fr'/mnt/pai-storage-1/tianyuan/workspace/smoke/sqyolov8/check/yolov8s_smoke_0716_2/{i}'

        # 确保目标目录存在
        os.makedirs(select_path, exist_ok=True)

        with open(txt_path, 'r') as f:
            data = [x.strip() for x in f.read().strip().splitlines() if len(x)]

        # 检查每一个入口
        for img_path in tqdm(data):
            # img2_path = os.path.join(source_path, os.path.split(img_path)[-1])
            # 确定目标路径
            target_path = os.path.join(select_path, os.path.split(img_path)[-1])
            # os.remove(img_path)
            shutil.move(img_path, target_path)
            # shutil.copy(img_path, target_path)
            # shutil.copy(source_path, target_path)






if __name__ == "__main__":
    n_class = 3
    k = 0
    img_select(n_class)
