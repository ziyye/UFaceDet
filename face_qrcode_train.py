import os
from ultralytics import YOLO

CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
TIMESTAMP = os.environ.get('TIMESTAMP', '0512')

# Load a model
model_path = r'./configs/yolov8-tiny2.yaml'
pt_file = r'./yolov8n.pt'
model = YOLO(model_path).load(pt_file)  # build a new model from scratch

# Train the model
dataset = r'./configs/face_qrcode2.yaml'
results = model.train(data=dataset,
                      epochs=400, 
                      imgsz=640, 
                      batch=256,
                      device=CUDA_VISIBLE_DEVICES,
                      workers=12, # 4*4090 取12，2*4090 取 10
                    #   single_cls = True,
                    #   freeze = [10],
                      # patience = 50,
                      optimizer = 'AdamW',
                      lr0 = 0.001,
                      weight_decay = 0.1,
                      close_mosaic = 10,
                      box = 3,
                      cls = 0.5,
                      amp = False,
                      save_period=1,  # Save checkpoint every epoch
                      project = r'./runs/detect',
                      name = f'8ntiny_640_face_qrcode_{TIMESTAMP}')

# export PYTHONPATH=/mnt/pai-storage-8/tianyuan/face_qrcode_det/yolov8:$PYTHONPATH
# nohup python face_qrcode_train.py > train.log 2>&1 &