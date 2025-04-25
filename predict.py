import glob
import os
import shutil
from ultralytics import YOLO


# Load a model
model = YOLO('/mnt/pai-storage-8/tianyuan/face_qrcode_det/yolov8/weights/best.pt')  # load an official model
# model = YOLO('/mnt/pai-storage-8/tianyuan/pfd/sqyolov8/runs/detect/person_face2_640_0119/weights/best.pt')  # load an official model
# smoke:/mnt/pai-storage-1/tianyuan/workspace/smoke/sqyolov8/weights/SP_1106_0821.pt
print("Model loaded.")
# Predict with the model
# imgs_dir = r'/mnt/pai-storage-8/tianyuan/facedet/sqyolov8/test_data/表情包'
imgs_dir = r'/mnt/pai-storage-12/data/Facedetect_data/FDDB/val/images'
# imgs_dir = r'/mnt/pai-storage-12/data/qrcode_data/qrcode/synthetise/val/images/00000001_sync_backflow.jpg'
# imgs_dir = r'/mnt/pai-storage-14/algorithm/zhouyanggang/datasets/Facedetect_data/yolo-face_data/Uface/val/images'
# imgs_dir = r'/mnt/pai-storage-12/data/animal/cats_vs_dogs/datasets--microsoft--cats_vs_dogs/images/'
# imgs_dir = r'/mnt/pai-storage-8/tianyuan/face_qrcode_det/sqyolov8/0421095818740_rgb.png'
# imgsList = glob.glob(imgs_dir)
# epoch = len(imgsList) // 1000 + 1
# for i in range(epoch):
#     imgs = imgsList[1000*i:1000*(i+1)]
results = model(imgs_dir,
                imgsz = (320,192),
                # imgsz = (288,480),
                # imgsz = 640,
                # batch= 4,
                conf = 0.5,
                iou = 0.45,
                # device = '0,1,2,3',
                # save_crop = True,
                # save = True,
                save_txt = True,
                # exist_ok = True,
                project = f'/mnt/pai-storage-8/tianyuan/face_qrcode_det/yolov8/runs',
                name = f'predict_pt/test_',
                )  # predict on an image
# print(results[0].boxes.data)
print('detect done!')

# check_dir = r'/mnt/pai-storage-1/tianyuan/workspace/facedet/sqyolov8/test_imgs/result_img'
# for result in results:
#     boxes = result.boxes
#     if boxes:
#         # for d in boxes:
#         #     c, conf, id = int(d.cls), float(d.conf) , None if d.id is None else int(d.id.item())
#             # source_dir = os.path.join(result.save_dir,os.path.split(result.path)[-1])
#             source_dir = result.path
#             target_dir = os.path.join(check_dir,os.path.split(result.path)[-1])
#             if os.path.exists(source_dir):
#                 shutil.copy(source_dir,target_dir)
            
""" 
imgsz:320 epoch:200 input:320
num_faces_gt=5171 tpr=95.55% fp=411 

imgsz:640 epoch:100 input:320
num_faces_gt=5171 tpr=96.19% fp=358

imgsz:640 epoch:200 input:320
num_faces_gt=5171 tpr=96.62% fp=427
"""      


