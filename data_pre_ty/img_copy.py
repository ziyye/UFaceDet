import os
import shutil
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

def move_images(source_dir, size, train_dir, train_num, val_dir=None, val_num=0):
    """
    将 source_dir 目录中的图片文件移动到 target_dir 目录
    """
    # 检查目标目录是否存在，如果不存在，创建它
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)   
    
    # 遍历源目录中的所有文件
    data_dir =os.listdir(source_dir)
    random.shuffle(data_dir)
    if train_num < 1:
        split_point = int(train_num * len(data_dir))
        train = data_dir[:split_point]
        val = data_dir[split_point:]
    else:
        train = data_dir[:train_num]
        val = data_dir[train_num:train_num+val_num]
    for filename in tqdm(train):
        # 构建完整的文件路径
        file_path = os.path.join(source_dir, filename)
        # 检查是否为文件
        if os.path.isfile(file_path):
            # 移动文件到目标目录
            im = cv2.imread(file_path)
            im2 = letterbox(im,size)[0]
            new_file_path = os.path.join(train_dir, filename)
            cv2.imwrite(new_file_path,im2)
            # shutil.copy(file_path, os.path.join(target_dir, filename))
            # print(f"Moving {filename} to {target_dir}")
        else:
            print(f"Skipping {filename}, it's not a file.")
    if val_dir:
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)
        for filename in tqdm(val):
            # 构建完整的文件路径
            file_path = os.path.join(source_dir, filename)
            # 检查是否为文件
            if os.path.isfile(file_path):
                # 移动文件到目标目录
                im = cv2.imread(file_path)
                im2 = letterbox(im,size)[0]
                new_file_path = os.path.join(val_dir, filename)
                cv2.imwrite(new_file_path,im2)
                # shutil.copy(file_path, os.path.join(target_dir, filename))
                # print(f"Moving {filename} to {target_dir}")
            else:
                print(f"Skipping {filename}, it's not a file.")

if __name__ == "__main__":
    # paths = r'/mnt/pai-storage-ceph-hdd/Dataset/Vehicle/QIDIAN/color_class/images_cls2'
    # # for key in os.listdir(paths):
    source_dir = fr"/mnt/pai-storage-1/tianyuan/workspace/facedet/sqyolov8/check/U_8ntiny_640_1202_u8"
    train_dir = fr"/mnt/pai-storage-1/tianyuan/workspace/facedet/sqyolov8/check/U_8ntiny_640_1202_u8_320_192"
    # val_dir = fr"/mnt/pai-storage-ceph-hdd/Dataset/Vehicle/QIDIAN/color_class/dataset/pos_cls{size}_3500/val/{key}"
    move_images(source_dir,(320,192), train_dir, 6000)
    print("All images have been moved successfully.")
    # for key in range(6):
    #         source_dir = fr"/mnt/pai-storage-ceph-hdd/Dataset/Vehicle/QIDIAN/color_class/dataset/pos_cls/val/{key}"
    #         train_dir = fr"/mnt/pai-storage-ceph-hdd/Dataset/Vehicle/QIDIAN/color_class/dataset/pos_cls416/val/{key}"
    #         # val_dir = fr"/mnt/pai-storage-ceph-hdd/Dataset/Vehicle/QIDIAN/color_class/dataset/pos_cls{size}/val/{key}"
    #         move_images(source_dir,416,train_dir,500)
    #         print("All images have been moved successfully.")