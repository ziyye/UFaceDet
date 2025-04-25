


"""
coco 数据集格式转化为 yolo.txt 数据集格式

"""

import os 
import json
import shutil
import cv2
import numpy as np
from tqdm import tqdm

classes = ['helmet','person','ebike']
def get_pic_id_map(annotation_file):
    """
    a function which get infomation from coco json file
    :param annotation_file: coco file
    :return:
        annotation_data:[{'file_name',   ;
            'height',   ;
            'width',   ;
            'anno_info', [[xmin, ymin, o_width, o_height, area, category_id],...]},
            ...]
        categories:all categories in the coco file
    """
    with open(annotation_file, 'r') as f:
        dataset = json.load(f)
    new_jsoninfo = {}
    if dataset.get('info'):
        new_jsoninfo['info'] = dataset['info']
    new_jsoninfo['categories'] = dataset['categories']
    annotation_data = []
    anno_box_data = {}
    # for key in range(len(dataset['images'])):
    #     anno_box_data[key] = []
    for img in dataset['images']:
        anno_box_data[img['id']] = []
    for anno in dataset['annotations']:
        key = anno['image_id']
        anno_info = anno['bbox']
        # seg = anno['segmentation'][0]
        # x1 = min(seg[::2])
        # y1 = min(seg[1::2])
        # x2 = max(seg[::2])
        # y2 = max(seg[1::2])
        # w = x2-x1
        # h = y2-y1
        # anno_info = [x1,y1,w,h]
        anno_info.append(int(anno['area']))
        anno_info.append(int(anno['category_id']))
        anno_box_data[key].append(anno_info)
    for image in dataset['images']:
        image_anno = {}   # 在循环里申明，重分配内存，防止值被覆盖
        image_anno['file_name'] = image['file_name']
        image_anno['height'] = image['height']
        image_anno['width'] = image['width']
        image_anno['anno_info'] = anno_box_data[image['id']]
        annotation_data.append(image_anno)
    new_jsoninfo['anno_info'] = annotation_data
    return new_jsoninfo
def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[2] / 2.0 )* dw
    y = (box[1] + box[3] / 2.0 )* dh
    w = box[2] * dw
    h = box[3] * dh
    return (x, y, w, h)

def convert1(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[2]) / 2.0 * dw
    y = (box[1] + box[3]) / 2.0 * dh
    w = (box[2] - box[0]) * dw
    h = (box[3] - box[1])* dh
    return (x, y, w, h)

def convert2(box0):
    box0 = np.array(box0)
    xmin,ymin = np.min(box0,0)
    xmax,ymax = np.max(box0,0)
    box= [xmin,ymin,xmax,ymax]
    return box

def coco2yolo(json_path,labels_path):
    data = json.load(open(json_path, 'r'))
    if not os.path.exists(labels_path):
        os.makedirs(labels_path)
    for img in tqdm(data):
        img_name = img["img_name"]
        # imgs_path = r'/mnt/pai-storage-ceph-hdd/Dataset/smoke_ty/data_huiliu/202312after/head_row/dataset/images'
        # new_path = os.path.join(imgs_path,img_name)
        img_data = cv2.imread(img["img_path"])
        h,w,_ = img_data.shape
        txt_name =os.path.splitext(img_name)[0] + '.txt'  # 对应的txt名字，与jpg一致
        if img["box_info"]:
            # shutil.copy(img["img_path"],new_path)
            with open(os.path.join(labels_path, txt_name), 'w') as out_file: 
                for ann_id in img["box_info"]:
                    cls = str(int(ann_id["box_type"]) - 1)
                    ann = ann_id['box']
                    box = convert1((w, h), ann)
                    # box = ann["bbox"]
                    out_file.write(f"{cls} {box[0]:.5f} {box[1]:.5f} {box[2]:.5f} {box[3]:.5f}\n")
if __name__ == '__main__':
    json_path = fr'/mnt/pai-storage-1/tianyuan/workspace/smoke/sqyolov8/select/json_files/smoke_det_0730.json'  # all_difficult x_all  COCO\annotation\merged_800_800.json
    labels_path = r'/mnt/pai-storage-ceph-hdd/Dataset/smoke_ty/data_huiliu/202312after/head_row/dataset/labels' # 保存的路径
    # coco2yolo2(json_path,labels_path)
    coco2yolo(json_path,labels_path)


        




