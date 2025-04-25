import glob
import json
import numpy as np
from random import shuffle
import os
import shutil
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm
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


# save_dir = '/mnt/pai-storage-8/tianyuan/rv1109_quant/smoke/quant_imgs'
save_dir = '/mnt/pai-storage-14/algorithm/zhouyanggang/datasets/Facedetect_data/yolo-face_data/animals/train/images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# data_dir = '/mnt/pai-storage-14/algorithm/zhouyanggang/datasets/Facedetect_data/yolo-face_data/Uface/train/images'
# # txt_path = '/mnt/pai-storage-1/tianyuan/workspace/code_common/rv1109_quant/facedet/Uface_test.txt'
# with open(txt_path,'r') as f:
#     for line in f:
#         data = json.loads(line.strip())
#         img_path2 = os.path.join(save_dir,os.path.split(data['image_path'])[-1][:-4]+'.png')
#         img = cv2.imread(os.path.join(data_dir,data['image_path']))
#         x1,y1,w,h = data['box']
#         x2 = x1 + w
#         y2 = y1 + h
#         crop_image = letterbox(img[y1:y2,x1:x2],[96,96])[0]
#         cv2.imwrite(img_path2,crop_image)
# print('ok')
# source_dir = '/mnt/pai-storage-1/tianyuan/workspace/facedet/sqyolov8/test_imgs/result_img'
# source_dir = '/mnt/pai-storage-8/tianyuan/smoke/data/dataset/warning_dataset_1/train/images/*'
source_dir = '/mnt/pai-storage-12/data/animal/cats_vs_dogs/datasets--microsoft--cats_vs_dogs/images/*'
# source_dir2 = '/mnt/pai-storage-8/tianyuan/pfd/sqyolov8/runs/detect/predict_pt/test_video9'
imgList = glob.glob(source_dir)
shuffle(imgList)
for img_path in tqdm(imgList[:3000]):
# for img_path in os.listdir(source_dir2):
    # img_path1 = os.path.join(source_dir,img_path)
    img_path2 = os.path.join(save_dir,os.path.split(img_path)[-1][:-4]+'.png')
    # img_path2 = os.path.join(save_dir,os.path.split(img_path)[-1])
    # img_path2 = os.path.join(save_dir,img_path)
    im = cv2.imread(img_path)
    im2 = letterbox(im,[416,416])[0]
    cv2.imwrite(img_path2,im2)