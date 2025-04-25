
import json
import numpy as np
import cv2
import os
import shutil
from tqdm import tqdm
import cv2
import xml.etree.ElementTree as ET
import random
import copy
# from check_pred import check_box
img_format = ['.jpg','.png','.tif','.JPEG']
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
        # anno_info = [x1,y1,x2-x1,y2-y1]
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
    categories = dataset['categories']
    return new_jsoninfo,categories

def get_pic_id_map2(annotation_file):
    with open(annotation_file, 'r') as f:
        dataset = json.load(f)
    anno_box_data = {}
    for anno in dataset['annotations']:
        img_name = anno['image_name']
        anno_box_data[img_name] = anno['bbox']
    return anno_box_data
def yolo2xywh(size, box):
    w = size[0]
    h = size[1]
    bx = int((box[0] - box[2] / 2) * w)
    by = int((box[1] - box[3] / 2) * h)
    bw = int(box[2]  * w)
    bh = int(box[3] * h)
    return (bx,by, bw, bh) 
 
def yolo2xywh2(size, box):
    w = size[0]
    h = size[1]
    box2 = [int(j*w)  if i % 2 == 0 else int(j*h) for i,j in enumerate(box)]
    box2 = np.array(box2).reshape(-1,2)
    return box2

def rec_vision1(txt_path,img_path):
    txt_list = os.listdir(txt_path)
    # txt_list.sort(key=lambda x: int(x[8:-4]),reverse = False) 
    random.shuffle(txt_list)
    for file in txt_list[:10]:
        txt_file = os.path.join(txt_path,file)
        for i in img_format:
            img_file = os.path.join(img_path,os.path.splitext(file)[0] + i)
            if os.path.exists(img_file):
                break
        if not os.path.exists(img_file):
            print(f'{img_file}图片格式不存在') 
            continue
        img0 = cv2.imread(img_file)
        h,w = img0.shape[:-1]
        # img0 = img0.astype(np.uint8)
        with open(txt_file, 'r') as f:
            data = [x.split() for x in f.read().strip().splitlines() if len(x)]
            for da in data:
                box0 = [eval(a) for a in da[1:5]]
                # points = [eval(a) for a in da[5:]]
                # box = [box0[0],box0[1],box0[2]-box0[0],box0[3]-box0[1]]
                # box = box0      
                cls = da[0]
                conf = 0.97
                label = f'{cls} {conf:.2f}'
                box = yolo2xywh((w,h),box0)
                # points2 = yolo2xywh2((w,h),points)
                # for point in points2:
                    # cv2.circle(img0, (int(point[0]),int(point[1])), 2, (0, 255, 0), 2)
                # cv2.polylines(img0,[points],True,(0,0,255),2)
                # cv2.rectangle(img0, (box[0], box[1]),(box[2], box[3]), 255, 2)
                cv2.rectangle(img0,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(0,255,0),2)
                # cv2.putText(img0, label, (box[0], box[1]+box[3]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)   
        cv2.imwrite(f'/mnt/pai-storage-1/tianyuan/workspace/facedet/sqyolov8/check/{file[:-4]}.jpg',img0)
def seg_vision1(txt_path,img_path):
    txt_list = os.listdir(txt_path)
    # txt_list.sort(key=lambda x: int(x[8:-4]),reverse = False) 
    random.shuffle(txt_list)
    for file in txt_list:
        cv2.namedWindow(f'{file}',0)
        txt_file = os.path.join(txt_path,file)
        for i in img_format:
            img_file = os.path.join(img_path,os.path.splitext(file)[0] + i)
            if os.path.exists(img_file):
                break
        img0 = cv2.imread(img_file)
        h,w = img0.shape[:-1]
        # img0 = img0.astype(np.uint8)
        with open(txt_file, 'r') as f:
            data = [x.strip() for x in f.read().strip().splitlines() if len(x)]
            for da in data:
                cls = da[0]
                box0 = np.fromstring(da[2:],sep = ' ').reshape(-1,2)*(w,h)
                box = box0.astype(int)
                cv2.polylines(img0,[box],True,(0,0,255),2)
                # cv2.rectangle(img0, (box[0], box[1]),(box[2], box[3]), 255, 2)
                # cv2.rectangle(img0,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(0,255,0),2)
                # cv2.putText(img0, label, (box[0], box[1]+box[3]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)   
        cv2.resizeWindow(f'{file}', int(w), int(h))                     
        cv2.imshow(f'{file}', img0)              
        key = cv2.waitKey(0)
        if key == 27:
            break
        cv2.destroyAllWindows()
def rec_vision2(json_path,img_path,imgs2_path):
    data, _ = get_pic_id_map(json_path)
    # random.shuffle(data['anno_info'])
    for ann in data['anno_info']:
        cv2.namedWindow(ann['file_name'])
        file_name = os.path.join(img_path,ann['file_name'])
        file2_name = os.path.join(imgs2_path,ann['file_name'])
        img0 = cv2.imread(file_name)
        # img0 = img0.astype(np.uint8)
        box_all = ann['anno_info']
        for box in box_all:
            box = [int(b) for b in box]
            # cv2.rectangle(img0,(box[0],box[1]),(box[2],box[3]),(0,255,0),1)
            cv2.rectangle(img0,(box[0],box[1]),(box[2]+box[0],box[3]+box[1]),(0,255,0),2)
            # cv2.putText(img0, str(box[5]), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow(ann['file_name'], img0) 
        key = cv2.waitKey(0)
        if key == 27:
            break
        cv2.destroyAllWindows()
        cv2.imwrite(file2_name,img0)
def rec_vision22(json_path,img_path):
    data = get_pic_id_map2(json_path)
    # random.shuffle(data['anno_info'])
    # cv2.namedWindow('check', cv2.WINDOW_NORMAL)
    img_list = os.listdir(img_path)
    # img_list.sort(key=lambda x: int(x[:-4])) 
    for img in img_list:
        file_name = os.path.join(img_path,img)
        img0 = cv2.imread(file_name)
        # img0 = img0.astype(np.uint8)
        for box in data[img]:
            cla = box['class']
            x,y,w,h = box['left'],box['top'],box['width'],box['height']
            cv2.rectangle(img0,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.putText(img0, str(cla), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow('check', img0)
        key = cv2.waitKey(30)
        if key == 27:
            break
        cv2.imwrite('')
    cv2.destroyAllWindows()
def rec_vision3(labels_path,imgs_path):
    # xml 标注可视化
    imgs_list = os.listdir(imgs_path)
    # imgs_list.sort(key=lambda x: int(x[:-4]))    
    labels_list = os.listdir(labels_path)
    cv2.namedWindow("ll", 0)
    # cv2.namedWindow("Obj", cv2.WINDOW_AUTOSIZE)
    for i in range(0, len(imgs_list)):
        filename = imgs_list[i]
        img_num = filename.split(".")[0]  # 获得图片的序号
        print("------------------------------------------------------------------------------")
        print("doing ", i, "-------", filename, "img.......")
        xml_file = img_num + ".xml"
        xml_path = os.path.join(labels_path,xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')            # 图片的shape值
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for obj in root.iter('object'):
            # difficult = obj.find('difficult').text
            cls_name = obj.find('name').text
            # if cls not in classes or int(difficult) == 1:
                # continue
            # 将名称转换为id下标
            # cls_id = classes.index(cls)
            # 获取整个bounding box框
            bndbox = obj.find('bndbox')
            # points = obj.find('points')
            # xml给出的是x1, y1, x2, y2
            box = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text), int(bndbox.find('xmax').text),
                int(bndbox.find('ymax').text)]
            x1,y1,x2,y2 = [box[0],box[1],box[2],box[3]]
            _image = cv2.imread(os.path.join(imgs_path, img_num + ".jpg"))
            # _image = image.copy()

            # obj_img = _image[y1:y2, x1:x2]
            # cv2.putText(_image, cls_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.rectangle(_image, (x1, y1), (x2, y2), (0, 0, 255), 2, 4)
            cv2.resizeWindow("ll", w, h)
            # cv2.imshow("Obj", obj_img)
            # cv2.waitKey()
            # print("当前目标的类别是：", cls_name)
            # print(x1, y1, x2, y2)
            # print("宽：", (x2 - x1), ",高：", (y2 - y1))
        cv2.imshow("ll", _image)
        cv2.waitKey() 
    cv2.destroyAllWindows()        
def video_vision(txt_file,video_path):
    fra_num = 0
    with open(txt_file, 'r') as f:
        data = [x.strip().split(' ') for x in f.read().strip().splitlines() if len(x)]
    frame_bbox = {}
    b = 0 
    for da in data:
        key = eval(da[5])
        box0 = [int(eval(a)) for a in da[1:5]]
        if not frame_bbox.get(key):
            frame_bbox[key] = []
        frame_bbox[key].append(box0)
    video = cv2.VideoCapture(video_path)
    prop = cv2.CAP_PROP_FRAME_COUNT
    total = int(video.get(prop))
    print("[INFO] 视频总帧数：{}".format(total))
    while True:
        ret,frame = video.read()
        if ret:
            for box in frame_bbox[fra_num]:
                x1,y1,x2,y2 = box
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
            fra_num += 1
            cv2.imshow('ll',frame)
            key = cv2.waitKey(25)
            if key == ord('q'):
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()
    print("[INFO] 当前视频帧数：{}".format(fra_num))
def video_vision2(txt_file,imgs_path):
    with open(txt_file, 'r') as f:
        data = [x.strip().split(',') for x in f.read().strip().splitlines() if len(x)]
    frame_bbox = {}
    for da in data:
        key = eval(da[0])
        box0 = [int(eval(a)) for a in da[1:6]] 
        if not frame_bbox.get(key):
            frame_bbox[key] = []
        frame_bbox[key].append(box0)
    frame_bbox = sorted(frame_bbox.items())
    frame_bbox = {k:v for k,v in frame_bbox}
    img_list = os.listdir(imgs_path)
    sorted(img_list,key = lambda x: int(x[3:-4]) )
    for id,img in enumerate(img_list):
        img_path = os.path.join(imgs_path,img)
        img0 = cv2.imread(img_path)    
        for box in frame_bbox[id+1]:
            cls,x1,y1,w,h = box
            cv2.rectangle(img0,(x1,y1),(x1+w,y1+h),(255,0,0),2)
            cv2.putText(img0, str(cls), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)                        
        cv2.imshow('ll',img0)
        key = cv2.waitKey(0)
        if key == 27:
            break
    cv2.destroyAllWindows()
def video_vision3(txt_path,video_path):
    txt_list = os.listdir(txt_path)
    txt_list.sort(key=lambda x: int(x[22:-4]),reverse = False)
    fra_num = 0
    frame_bbox = {}
    for file in txt_list: 
        txt_file = os.path.join(txt_path,file)
        key = int(file[22:-4])
        if not frame_bbox.get(key):
            frame_bbox[key] = []
        with open(txt_file, 'r') as f:
            data = [x.strip() for x in f.read().strip().splitlines() if len(x)]   
            for da in data:
                cls = da[0]
                box0 = np.fromstring(da[2:],sep = ' ').reshape(-1,2)*np.array([800,1360])
                box = box0.astype(int)                  
                frame_bbox[key].append(box)
    video = cv2.VideoCapture(video_path)
    prop = cv2.CAP_PROP_FRAME_COUNT
    total = int(video.get(prop))
    print("[INFO] 视频总帧数：{}".format(total))
    txt_list = []
    cv2.namedWindow('ll',cv2.WINDOW_FREERATIO)
    while True:
        ret,frame = video.read()
        if ret:
            print(fra_num)
            print(frame.shape)                
            fra_num += 1      
            frame2 = copy.deepcopy(frame)
            if frame_bbox.get(fra_num):
                for box in frame_bbox[fra_num]:
                    cv2.polylines(frame2,[box],True,(0,0,255),2)
            cv2.imshow(f'll',frame2)
            key = cv2.waitKey()
            # cv2.waitKey(25)
            if key == 27:              
                break
            if key == ord('w'):  
                txt_list.append(fra_num)
                cv2.imwrite(fr'D:\workspace\datasets\bridge_crack2\1\images\{fra_num}.jpg',frame)
        else:
            break
    video.release()
    print("[INFO] 当前视频帧数：{}".format(fra_num))
    print(txt_list)
if __name__ == '__main__':
    # root = r'D:/FDDB'
    # imgs_path = fr'{root}\images'
    # txt_path = fr'{root}\labels'
    # txt_path = fr'{root}\Annotations'
    # json_path = fr'{root}\best_predictions.json'  # all_difficult x_all  COCO\annotation\merged_800_800.json
              
    imgs_path = r'/mnt/pai-storage-ceph-hdd-nfs/Dataset/FDDB/images'
    txt_path = r'/mnt/pai-storage-1/tianyuan/workspace/facedet/yolov5-face/runs/detect/exp/labels'
    # txt_path = fr'C:/Users/tianyuan/Downloads/test/labels' 
    # imgs_path = r'C:/Users/tianyuan/Downloads/test/images'
    # imgs_path = r'C:/Users/tianyuan/Downloads/imgs'
    # imgs2_path = fr'E:\workspace\test_data\DOTA100'
    # json_path = fr'E:\workspace\test_data\DOTA100\new_all.json'
    # video_path = fr'D:\workspace\sqyolov8\test_imgs\vid\70_15020231219_153653.MKV'
    # txt 标注可视化
    # save_path = 'E:/workspace/DataProcess/1'
    rec_vision1(txt_path,imgs_path)
    # seg_vision1(txt_path,imgs_path)
    # json 标注可视化
    # rec_vision2(json_path,imgs_path,imgs2_path)
    # video_vision2(txt_path,imgs_path)
    # video_vision3(txt_path,video_path)
    # rec_vision3(txt_path,imgs_path)


