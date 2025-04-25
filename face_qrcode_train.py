
from ultralytics import YOLO
# Load a model
model_path = r'./configs/yolov8-tiny2.yaml'
pt_file = r'./yolov8n.pt'
model = YOLO(model_path).load(pt_file)  # build a new model from scratch

# Train the model
dataset = r'./configs/face_qrcode2.yaml'
results = model.train(data=dataset,
                      epochs=200, 
                      imgsz=640, 
                      batch=96,
                      device='1,2,3',
                      workers=8, # 4*4090 取12，2*4090 取 10
                    #   single_cls = True,
                    #   freeze = [10],
                      patience = 30,
                      optimizer = 'AdamW',
                      lr0 = 0.001,
                      weight_decay = 0.1,
                      close_mosaic = 10,
                      box = 3,
                      cls = 0.5,
                      amp = False,
                      project = r'/mnt/pai-storage-8/tianyuan/face_qrcode_det/yolov8/runs/detect',
                      name = '8ntiny_640_face_qrcode_0424')

# export PYTHONPATH=/mnt/pai-storage-8/tianyuan/face_qrcode_det/yolov8:$PYTHONPATH
# nohup python face_qrcode_train.py > train.log 2>&1 &