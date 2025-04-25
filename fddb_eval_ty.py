import glob
import cv2
import os
import numpy as np
import torch
from tqdm import tqdm
# from detect import CenterFace


def yolo2xywh(size, box):
    w = size[0]
    h = size[1]
    bx = int((box[0] - box[2] / 2) * w)
    by = int((box[1] - box[3] / 2) * h)
    bw = int(box[2]  * w)
    bh = int(box[3] * h)
    return (bx,by, bw, bh) 

def xyxy2xywh(box):
    bx = int(box[0])
    by = int(box[1])
    bw = int(box[2] - box[0])
    bh = int(box[3] - box[1])
    return (bx,by, bw, bh)

def plot_boxes(img, boxes, filepath=None, color=(0, 255, 0)):
    boxes_show = boxes.astype(np.int32)
    for i in range(len(boxes_show)):
        cv2.rectangle(img, (boxes_show[i, 0], boxes_show[i, 1]),
                      (boxes_show[i, 0] + boxes_show[i, 2],
                       boxes_show[i, 1] + boxes_show[i, 3]),
                      color, thickness=2)
        # cv2.putText(img, str(i), (boxes_show[i, 0], boxes_show[i, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if filepath is not None:
        cv2.imwrite(filepath, img)
    return img



def load_data(result_label_dir, labels_dir, images_dir):
    
    data_all = []

    labels_List = glob.glob(labels_dir)
    for label_file in tqdm(labels_List):
        img_path = os.path.join(images_dir,os.path.split(label_file)[1][:-4] + '.jpg')
        # img_path = os.path.join('/mnt/pai-storage-14/algorithm/zhouyanggang/datasets/Facedetect_data/yolo-face_data/Uface/val/images',os.path.split(label_file)[1][:-4] + '.jpg')
        # img_path = os.path.join('/mnt/pai-storage-8/tianyuan/pfd/data/normal/val/images',os.path.split(label_file)[1][:-4] + '.jpg')    
        # img_path_jpeg = os.path.join('/mnt/pai-storage-8/tianyuan/pfd/data/normal/val/images',os.path.split(label_file)[1][:-4] + '.jpeg')
        
        img = cv2.imread(img_path)
        # if img is None:
        #     img = cv2.imread(img_path_jpeg)
        #     img_path = img_path_jpeg
        # if img is None:
        #     print(f"无法读取图像: {img_path} 或 {img_path_jpeg}")
        #     continue
        
        h, w = img.shape[:-1]
        boxes_gt = []
        with open(label_file, 'r') as f:
            data_gt = [x.split() for x in f.read().strip().splitlines() if x.startswith('0')]
            if len(data_gt) == 0:
                continue
            for da_gt in data_gt:
                box0_gt = [eval(a) for a in da_gt[1:5]]
                box_gt = yolo2xywh((w,h),box0_gt)
                boxes_gt.extend(box_gt)    
        boxes_gt = np.array(boxes_gt, dtype=np.float32).reshape(-1, 4)

        label_file2 = os.path.join(result_label_dir,os.path.split(label_file)[1])
        boxes = []
        if os.path.exists(label_file2):
            with open(label_file2, 'r') as f:
                data_pred = [x.split() for x in f.read().strip().splitlines() if len(x)]
                for da_pred in data_pred:
                    if len(da_pred) == 5:
                        box0_pred = [eval(a) for a in da_pred[1:5]]
                        box_pred = yolo2xywh((w,h),box0_pred)
                        # box_pred = xyxy2xywh(box0_pred)
                    else:
                        box0_pred = [eval(a) for a in da_pred[0:4]]
                        box_pred = [box0_pred[0],box0_pred[1],box0_pred[2]-box0_pred[0],box0_pred[3]-box0_pred[1]]
                    boxes.extend(box_pred)    
            boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        else:
            boxes = np.array(boxes)
        # 将数据添加到列表中
        data_all.append({
            'img_path': img_path,
            'boxes_gt': boxes_gt,
            'boxes_pred':boxes,
            'image': img,  # 加载的图像
        })

    return data_all

def IoU(box, boxes):
    # box type: x, y, w, h
    box_area = box[2] * box[3]
    area = boxes[:, 2] * boxes[:, 3]
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2] + box[0] - 1, boxes[:, 2] + boxes[:, 0] - 1)
    yy2 = np.minimum(box[3] + box[1] - 1, boxes[:, 3] + boxes[:, 1] - 1)

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr_union = inter / (box_area + area - inter)
    # ovr_min = inter / np.min(box_area, area)
    return ovr_union


def eval_fddb(fddb_data, iou_ths=0.2, error_path=None):
    if error_path is not None:
        if os.path.exists(error_path):
            os.system('rm -r %s' % error_path)
        os.system('mkdir %s' % error_path)

    tp, fp = 0, 0
    num_faces = 0

    for i, item in enumerate(fddb_data):
        if (i + 1) % 200 == 0:
            print('%d/%d ...' % ((i + 1), len(fddb_data)))
        img_path = item['img_path']
        basename = os.path.split(img_path)[-1]
        I = item['image']
        boxes_gt = item['boxes_gt']
        num_faces += len(boxes_gt)
        boxes = item['boxes_pred']
            
        if boxes is None or len(boxes) == 0:
            plot_boxes(I, boxes_gt,
                       os.path.join(error_path, basename),
                       color=(255, 0, 0))
            continue

        tp_current, fp_current = 0, 0
        # print('fhfhfhfhhfhf', boxes_pred.shape)
        for box in boxes:
            Iou = IoU(box, boxes_gt)
            if np.max(Iou) > iou_ths:
                tp_current += 1
            else:
                fp_current += 1

        tp += tp_current
        fp += fp_current
        # if error_path is not None and (tp_current != len(boxes_gt) or fp_current > 0):
        if error_path is not None and (fp_current > 0):
            I = plot_boxes(I, boxes_gt, color=(0, 255, 0))
            plot_boxes(I, boxes,
                       os.path.join(error_path, basename),
                       color=(0, 0, 255))
        

    tpr = tp / num_faces
    print('num_faces_gt=%d tpr=%0.2f%%@fp=%d' % (num_faces, 100 * tpr, fp))
 

if __name__ == '__main__':
    fddb_dir = "/mnt/pai-storage-1/tianyuan/workspace/facedet/FDDB"
    # fddb_dir = "/mnt/pai-storage-14/algorithm/zhouyanggang/datasets/Facedetect_data/yolo-face_data/Uface"
    # fddb_dir = "/mnt/pai-storage-8/tianyuan/pfd/data/normal"
    result_labels_dir = r'/mnt/pai-storage-8/tianyuan/face_qrcode_det/yolov8/runs/predict_pt/test_/labels'
    error_dir = r'/mnt/pai-storage-8/tianyuan/face_qrcode_det/yolov8/check'
    labels_dir = os.path.join(fddb_dir, "val/labels/*")
    images_dir = os.path.join(fddb_dir, "val/images")
    fddb_data = load_data(result_labels_dir, labels_dir, images_dir)
    predict_error = os.path.join(error_dir, "fddb_0424_conf05_dv500")
    eval_fddb(fddb_data, error_path=predict_error)


'''

2002_07_26_big_img_936.txt
2003_01_14_big_img_660.txt
tpr=86.23%@fp=35
'''
