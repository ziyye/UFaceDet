import glob
import cv2
import os
import numpy as np
import torch
from tqdm import tqdm
import time
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



def load_data(pred_result_dir, gt_labels_dir, images_dir, class_idx):
    
    data_all = []

    labels_List = glob.glob(gt_labels_dir)
    for label_file in tqdm(labels_List, desc='loading GT data'):
        img_path = os.path.join(images_dir,os.path.split(label_file)[1][:-4] + '.jpg')
        
        img = cv2.imread(img_path)
        
        h, w = img.shape[:-1]
        boxes_gt = []
        with open(label_file, 'r') as f:
            data_gt = [x.split() for x in f.read().strip().splitlines() if x.startswith(str(class_idx) + ' ')]  # only evaluate one class at a time
            if len(data_gt) == 0:
                continue
            for da_gt in data_gt:
                box0_gt = [eval(a) for a in da_gt[1:5]]
                box_gt = yolo2xywh((w,h),box0_gt)
                boxes_gt.extend(box_gt)    
        boxes_gt = np.array(boxes_gt, dtype=np.float32).reshape(-1, 4)

        label_file2 = os.path.join(pred_result_dir,os.path.split(label_file)[1])
        boxes = []
        if os.path.exists(label_file2):
            with open(label_file2, 'r') as f:
                data_pred = [x.split() for x in f.read().strip().splitlines() if len(x)]
                for da_pred in data_pred:
                    if len(da_pred) == 5 and da_pred[0] == str(class_idx):
                        box0_pred = [eval(a) for a in da_pred[1:5]]
                        if box0_pred:
                            box_pred = yolo2xywh((w,h),box0_pred)
                        else:
                            continue
                        # box_pred = xyxy2xywh(box0_pred)
                    else:
                        box0_pred = [eval(a) for a in da_pred[0:4]]
                        box_pred = [box0_pred[0],box0_pred[1],box0_pred[2]-box0_pred[0],box0_pred[3]-box0_pred[1]]
                    boxes.extend(box_pred)    
            boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        else:
            boxes = np.array(boxes)
        
        data_all.append({
            'img_path': img_path,
            'boxes_gt': boxes_gt,
            'boxes_pred':boxes,
            'image': img,
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


def eval_testset(objects_data, iou_ths=0.2, error_path=None):
    if error_path is not None:
        if os.path.exists(error_path):
            os.system('rm -r %s' % error_path)
        os.system('mkdir %s' % error_path)

    if SAVE_TRUE_POSITIVE_BBOXES:
        correct_pred_bboxes_path = os.path.join(os.path.dirname(error_path), 'correct')
        os.system('mkdir %s' % correct_pred_bboxes_path)

    tp, fp = 0, 0
    num_objects_gt = 0

    for i, item in enumerate(objects_data):
        if (i + 1) % 200 == 0:
            print('%d/%d ...' % ((i + 1), len(objects_data)))
        img_path = item['img_path']
        basename = os.path.split(img_path)[-1]

        I = item['image']
        boxes_gt = item['boxes_gt']
        num_objects_gt += len(boxes_gt)
        boxes_pred = item['boxes_pred']  # if negative in boxes_pred, seems cv2.rectangle can not draw it right

        if boxes_pred is None or len(boxes_pred) == 0:
            plot_boxes(I, boxes_gt,
                       os.path.join(error_path, basename),
                       color=(255, 0, 0))  # blue
            continue

        tp_current, fp_current = 0, 0
        # print('fhfhfhfhhfhf', boxes_pred.shape)
        for box_pred in boxes_pred:
            Iou = IoU(box_pred, boxes_gt)
            if np.max(Iou) > iou_ths:
                tp_current += 1
            else:
                fp_current += 1

        tp += tp_current
        fp += fp_current
        # if error_path is not None and (tp_current != len(boxes_gt) or fp_current > 0):
        if error_path is not None and (fp_current > 0):
            I = plot_boxes(I, boxes_gt, color=(0, 255, 0))  # green
            plot_boxes(I, boxes_pred,
                       os.path.join(error_path, basename),
                       color=(0, 0, 255))  # red
        
        if SAVE_TRUE_POSITIVE_BBOXES and tp_current > 0 and fp_current == 0:
            I = plot_boxes(I, boxes_gt, color=(0, 255, 0))  # green
            plot_boxes(I, boxes_pred,
                        os.path.join(correct_pred_bboxes_path, basename),
                        color=(0, 0, 255))  # red


    tpr = tp / num_objects_gt
    print(f"[{TEST_SET_NAME}] num_objects_gt: {num_objects_gt}, tp: {tp}, tpr: {tpr*100:.2f}%, fp: {fp}")


if __name__ == '__main__':
    SAVE_TRUE_POSITIVE_BBOXES = True

    pred_results_dir = "/data/wangjiazhi/projs/yolov8_0512/runs/detect/8ntiny_640_face_qrcode_20250519_013541/predict_result/qrcode0512-epoch272-best/labels"

    # TEST_SET_NAME = "FDDB"  # FDDB, qrcode0512
    TEST_SET_NAME = "qrcode0512"

    testset_configs = {
        "FDDB": {
            "images_dir": "/mnt/pai-storage-12/data/Facedetect_data/FDDB/val/images",
            "labels_dir": "/mnt/pai-storage-12/data/Facedetect_data/FDDB/val/labels/*",
            "class_idx": 0  # 0 is face, 1 is qrcode
        },
        "qrcode0512": {
            "images_dir": "/mnt/pai-storage-12/data/qrcode_data/qrcode/250512/test/images",
            "labels_dir": "/mnt/pai-storage-12/data/qrcode_data/qrcode/250512/test/labels/*",
            "class_idx": 1  # 0 is face, 1 is qrcode
        }
    }
    testset_config = testset_configs[TEST_SET_NAME]
    labels_dir = testset_config["labels_dir"]
    images_dir = testset_config["images_dir"]
    class_idx = testset_config["class_idx"]
    
    visualize_dir = os.path.join(os.path.dirname(pred_results_dir), "check")
    visualize_error_dir = os.path.join(visualize_dir, "errors")

    print(f"visualize_error_dir: {visualize_error_dir}")
    if not os.path.exists(visualize_error_dir):
        os.makedirs(visualize_error_dir)
    else:
        raise ValueError(f"visualize_error_dir: {visualize_error_dir} already exists!")

    t0 = time.time()
    objects_data = load_data(pred_results_dir, labels_dir, images_dir, class_idx)
    t1 = time.time()
    print(f"Load data time cost: {(t1 - t0)/60:.2f} minutes")
    eval_testset(objects_data, error_path=visualize_error_dir)
    print(f"Eval time cost: {(time.time() - t1)/60:.2f} minutes")
