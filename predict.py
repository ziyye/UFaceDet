import os
import time
from ultralytics import YOLO

model_path = '/data/wangjiazhi/projs/yolov8_0512/runs/detect/8ntiny_640_face_qrcode_20250519_013541/weights/best.pt'

# TEST_SET_NAME = "FDDB"  # FDDB, qrcode0512
TEST_SET_NAME = "qrcode0512"  # FDDB, qrcode0512

result_save_proj = '/data/wangjiazhi/projs/yolov8_0512/runs'
result_save_name = f'detect/8ntiny_640_face_qrcode_20250519_013541/predict_result/{TEST_SET_NAME}-epoch272-best'
assert not os.path.exists(os.path.join(result_save_proj, result_save_name)), f"Result save path {result_save_name} already exists!"

model = YOLO(model_path)
print(f"Model loaded: {model_path}")

if TEST_SET_NAME == "FDDB":
    imgs_dir = r'/mnt/pai-storage-12/data/Facedetect_data/FDDB/val/images'
elif TEST_SET_NAME == "qrcode0512":
    imgs_dir = r'/mnt/pai-storage-12/data/qrcode_data/qrcode/250512/test/images'
else:
    raise ValueError(f"Invalid test set name: {TEST_SET_NAME}")
print(f"Testset {TEST_SET_NAME} loaded: {imgs_dir}")

start_time = time.time()
results = model(imgs_dir,
                imgsz = (320,192),
                # imgsz = (288,480),
                # imgsz = 640,
                # batch= 4,
                conf = 0.5,
                iou = 0.45,
                device = '4,5',
                # save_crop = True,
                # save = True,
                save_txt = True,
                # exist_ok = True,
                project = result_save_proj,
                name = result_save_name,  # predicted results will be saved in {project}/{name}/labels
                )  # predict on an image
# print(results[0].boxes.data)
print(f'Detect done! Time cost: {(time.time() - start_time)/60:.2f} minutes')


# step1: python predict.py
# step2: python eval_qrcode_face.py