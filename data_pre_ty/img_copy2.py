import glob
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

if __name__ == "__main__":
    # dms_list = ['0_normal','1_smoke','2_phone','3_smoke_phone']
    # dms_list = ['0_normal','1_smoke','2_phone']
    # for key in range(3):
    # for key in dms_list:
        # key = '1_smoke'
        # source_dir = fr"/mnt/pai-storage-1/tianyuan/workspace/smoke/yolov10/check/yolov8s-resnet101_smoke_3/1/*"
        source_dir = fr"/mnt/pai-storage-14/algorithm/zhouyanggang/datasets/Facedetect_data/yolo-face_data/Uface/val/images/*"
        # source_dir = fr"/mnt/pai-storage-1/tianyuan/workspace/smoke/sqyolov8/test_imgs/test0816/*"
        train_dir = fr"/mnt/pai-storage-1/tianyuan/workspace/facedet/sqyolov8/Uface_320_192"
        # train_dir = fr"/mnt/pai-storage-1/data/smoke_ty/dataset/train/{key}"
        # train_dir = fr"/mnt/pai-storage-1/tianyuan/workspace/smoke/sqyolov8/test_imgs/test08162"
        # val_dir = fr"/mnt/pai-storage-ceph-hdd/Dataset/smoke_ty/data_huiliu/dataset/val/{key}"
        # val_dir = fr"/mnt/pai-storage-ceph-hdd/Dataset/Vehicle/QIDIAN/pos_class/dataset/pos_cls_30pz/val/3"
        # 'smoke_20210706144628_RGB_1174243_47_579_384'
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)  
        # if not os.path.exists(val_dir):
        #     os.makedirs(val_dir)

        # imgList = []
        imgList = glob.glob(source_dir)
        # imgList2 = [x for x in imgList if os.path.split(x)[-1].startswith('smoke_202107')]
        random.shuffle(imgList)

        # train = imgList[:10]
        train = imgList[:10]
        # val = imgList[2:4]
        # train = imgList[:400]
        # val = imgList[-40:]

        for img_path in tqdm(train):
            img_path2 = os.path.join(train_dir,os.path.split(img_path)[-1][:-4]+'.png')
            if os.path.isfile(img_path):
                # 移动文件到目标目录
                im = cv2.imread(img_path)
                # im2 = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im2 = letterbox(im,[320,192])[0]
                cv2.imwrite(img_path2,im2)
                # shutil.move(img_path,img_path2)
                # shutil.copy(img_path,img_path2)
        # for img_path in tqdm(val):
        #     img_path2 = os.path.join(val_dir,os.path.split(img_path)[-1])
        #     if os.path.isfile(img_path):
        # #         # 移动文件到目标目录
        # #         im = cv2.imread(img_path)
        # #         im2 = letterbox(im,416)[0]
        # #         cv2.imwrite(img_path2,im2)
        #         shutil.copy(img_path,img_path2)
    # img_path = r'/mnt/pai-storage-1/tianyuan/workspace/smoke/sqyolov8/test_imgs/107_head0.jpg'
    # img_path2 = r'/mnt/pai-storage-1/tianyuan/workspace/smoke/sqyolov8/test_imgs/head/107_head0.jpg'
    # if os.path.isfile(img_path):
    #     # 移动文件到目标目录
    #     im = cv2.imread(img_path)
    #     im2 = letterbox(im,256)[0]
    #     cv2.imwrite(img_path2,im2)    


            