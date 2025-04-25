from ultralytics import YOLO
import onnx
# Load a model
model_path = r'/mnt/pai-storage-8/tianyuan/face_qrcode_det/yolov8/runs/detect/8ntiny_640_face_qrcode_04212/weights/best.pt'
# model_path = r'/mnt/pai-storage-8/tianyuan/pfd/sqyolov8/weights/base/8n2_640_0107/weights/best.pt'
# model_path = r'/mnt/pai-storage-1/tianyuan/workspace/facedet/sqyolov8/runs/detect/debug_/weights/best.pt'
# model_path = r'/mnt/pai-storage-1/tianyuan/workspace/smoke/sqyolov8/runs/detect/smoke_8n416_shuffle_0905_/weights/best.pt'
model = YOLO(model_path)  # load a custom model

# Export the model
model.export(format='onnx',
             opset = 11,
             simplify = True,
            #  imgsz = [384,640],
             imgsz = [320,192],
            #  imgsz = 640,
            #  dynamic = True 
            )


cvt_model = onnx.load(model_path.replace('.pt','.onnx'))

idx_start = 0 
for output in cvt_model.graph.output:
    for node in cvt_model.graph.node:
        # 如果当前节点的输入名称与待修改的名称相同，则将其替换为新名称
        for i, name in enumerate(node.output):
            if name == output.name:
                node.output[i] = "out" + str(idx_start)
    output.name = "out" + str(idx_start)
    idx_start += 1

# 保存修改后的模型
onnx.save(cvt_model, model_path.replace('.pt','.onnx'))