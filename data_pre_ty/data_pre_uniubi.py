import glob
import os
import shutil
from tqdm import tqdm
import cv2
import json
from ultralytics import YOLO

class_name = ['smoke', 'phone']
def json_dump(json_path, results):
    with open(json_path, 'w', encoding='utf-8') as fw:
        data = []
        for result in tqdm(results):
            try:
                imgPath = result.path
                # image = cv2.imread(imgPath)
                # height, width = image.shape[:2]
                img_name = os.path.basename(imgPath)
                absolute_path = os.path.abspath(imgPath)
                img_data = {
                            "img_name": img_name,
                            "img_path": imgPath,
                            "box_info": [],
                            "point_info": []
                            }
                boxes = result.boxes.data.cpu().numpy()
                if len(boxes) != 0:
                    for box in boxes:
                        box = box.tolist()
                        x_min, y_min, x_max, y_max, conf, label_id = box
                        label = class_name[int(label_id)]  # 类别
                        temp_data = {"box_type": label, "box": [int(x_min), int(y_min), int(x_max), int(y_max)]}
                        # print(temp_data)
                        img_data["box_info"].append(temp_data)
                    data.append(img_data)
            except:
                print(imgPath)
                # os.remove(imgPath)
        json.dump(data, fw, ensure_ascii=False, indent=4)
if __name__ == '__main__':
    # img_dir_list = [
    #                 # '/mnt/pai-storage-ceph-hdd-nfs/Dataset/chache_backflow/2408-240906/images/*',
    #                 # '/mnt/pai-storage-1/tianyuan/workspace/smoke/sqyolov8/test_imgs/test/train/*',
    #                 '/mnt/pai-storage-8/tianyuan/smoke/data/vllm_predict/data/存在吸烟打电话行为/*',
    #                 ]
    img_dir = '/mnt/pai-storage-8/tianyuan/smoke/data/already_cleaned/smoke_phone'
  
    # imgLists = []
    # for img_dir in img_dir_list:
    #     imgLists += glob.glob(img_dir)
    # imgLists = list(filter(lambda path: os.path.splitext(path)[-1].lower() in [".jpg", ".png", ".jpeg"], imgLists))

    model_path = r'/mnt/pai-storage-1/tianyuan/workspace/smoke/sqyolov8/weights/SP_1106_0821.pt'
    model = YOLO(model_path)  # 权重

    json_path = '/mnt/pai-storage-8/tianyuan/smoke/data/already_cleaned/smoke_pre_0119.json'
    results = model.predict(img_dir, imgsz=416, iou=0.5, save=False, save_txt=False) 
    # print(len(imgLists))
    json_dump(json_path,results)