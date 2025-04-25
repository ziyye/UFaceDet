import shutil
import cv2
import numpy as np
import itertools
import glob
import os 
import json
import random
from tqdm import tqdm

import torch
import onnxruntime
import numpy as np
import torch.nn as nn
import torchvision
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
# 类别
det_classes = {0: 'color'}
color_classes = {0: 'bgry',1: 'bryg',2: 'byrg',3: 'gbry',4: 'gbyr',5: 'gybr',6: 'gyrb',7: 'rbgy',8: 'rbyg',9: 'rgby',10: 'rybg',11: 'rygb'}
# pos_classes = {0: '正常停放', 1: '车身靠前', 2: '车身靠后', 3: '车身偏置', 4: '车身倒置', 5: '无法判断'}
pos_classes = {0: 'Right', 1: 'Forward', 2: 'Backward', 3: 'Misaligned', 4: 'Inverted', 5: 'Indeterminable'}

def pre_process(img_path,size):  
    img_mat = cv2.imread(img_path)
    im0 , r, (dw,dh) = letterbox(img_mat,size)
    im = im0.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)
    im = im[np.newaxis, :].astype(np.float32)
    im /= 255
    # im = torch.from_numpy(im,dtype = torch.float)
    return im

def letterbox(im, new_shape=(416, 416), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))
 
    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name
 
    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name
 
    def get_input_feed(self, input_name, image_tensor):
        """
        input_feed={self.input_name: image_tensor}
        :param input_name:
        :param image_tensor:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_tensor
        return input_feed
 
    def forward(self, image_tensor):
        '''
        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor[np.newaxis, :]
        onnx_session.run([output_name], {input_name: x})
        :param image_tensor:
        :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_tensor})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: image_tensor})
        input_feed = self.get_input_feed(self.input_name, image_tensor)
        result= self.onnx_session.run(self.output_name, input_feed=input_feed)
        return result

def post_process(x,stride = torch.tensor([8,16,32],dtype=torch.float32)):
    
    x_cat = torch.cat([xi.view(1, 65, -1) for xi in x], 2)
    # x_cat = torch.cat([xi.view(shape0, shape1, -1) for xi in x], 2)
    box, cls = x_cat.split((16 * 4, 1), 1)
    anchors, strides = (x.transpose(0, 1) for x in make_anchors(x, stride, 0.5))
    dbox = dist2bbox(dfl(box), anchors.unsqueeze(0), xywh=True, dim=1) * strides
    y = torch.cat((dbox, cls.sigmoid()), 1)

    # onnx_paths = r'C:/Users/tianyuan/Downloads/qidian'
    # name = 'cls.txt'
    # data = cls.sigmoid()
    # cls_txt = np.array(data)
    # onnx_txt = os.path.join(onnx_paths,name)
    # np.savetxt(onnx_txt,cls_txt.reshape(1,-1),fmt='%1.6f', delimiter=' ')
    return y

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def dfl(x):
    c1 = 16
    conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
    w = torch.arange(c1, dtype=torch.float)
    conv.weight.data[:] = nn.Parameter(w.view(1, c1, 1, 1))
    b, c, a = x.shape  # batch, channels, anchors
    return conv(x.view(b, 4, c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)

def non_max_suppression(
        prediction,
        conf_thres=0.5,
        iou_thres=0.45,
        nc=1,  # number of classes (optional)
        max_wh=7680,
):
    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates
    # xd = prediction[:, 4:mi].tolist()
    # xd = [f'{x:.3f}' for x in xd[0][0]]
    # print(xd)
    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence
        # Cat apriori labels if autolabelling
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)
        conf, j = cls.max(1, keepdim=True)
        x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue

        # Batched NMS
        c = x[:, 5:6] *  max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres) 
        # i = i[:1]
        output[xi] = x[i]
    return output

def xywh2xyxy(x):
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    dw = x[..., 2] / 2  # half-width
    dh = x[..., 3] / 2  # half-height
    y[..., 0] = x[..., 0] - dw  # top left x
    y[..., 1] = x[..., 1] - dh  # top left y
    y[..., 2] = x[..., 0] + dw  # bottom right x
    y[..., 3] = x[..., 1] + dh  # bottom right y
    return y


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True):
   
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1), round(
            (img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def clip_boxes(boxes, shape):

    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
 


if __name__ == "__main__":

    pos_data = {}
    for i in range(6):
        pos_data[i] = []
    json_path = r'/mnt/pai-storage-ceph-hdd/Dataset/Vehicle/QIDIAN/pos_class/pos_crop_0523/crops/pos.json'
    # json_path = r'/mnt/pai-storage-1/tianyuan/workspace/qidian/sqyolov8/test_imgs/pos.json'

    img_dir_list = [
        f'/mnt/pai-storage-ceph-hdd/Dataset/Vehicle/QIDIAN/pos_class/pos_crop_0523/crops/color/*',
    ]
    # img_dir_list = [
    #     f'/mnt/pai-storage-1/tianyuan/workspace/qidian/sqyolov8/test_imgs/pos/*',
    # ]

    imgLists = []
    for img_dir in img_dir_list:
        imgLists += glob.glob(img_dir)
    imgLists = list(filter(lambda path: os.path.splitext(path)[-1].lower() in ['.jpg', '.jpeg', '.png'], imgLists))

    # random.shuffle(imgLists)
    # copy_path = r'/mnt/pai-storage-1/tianyuan/workspace/qidian/sqyolov8/test_imgs/cls2'
    # shutil.copy(image_path,os.path.join(copy_path,os.path.split(image_path)[-1]))

    onnx_path = '/mnt/pai-storage-1/tianyuan/workspace/qidian/sqyolov8/weights/yolov8n_416_v9.onnx'
    onnx_test = ONNXModel(onnx_path) 

    for image_path in tqdm(imgLists):

        # 前处理
        im = pre_process(image_path,416)
        im_tensour = torch.tensor(im,dtype=torch.float)
        # 模型推理
        onnx_result = onnx_test.forward(im)
    
        cls = onnx_result[0].argmax(1)[0]
        pos_data[cls].append(image_path)
        # prob = onnx_result[0].max(1)[0]
        # result = pos_classes[cls]
        # print(f'此图的类别是 {result}, 置信度为 {prob:.6f}')
    with open(json_path, 'w', encoding='utf-8') as fw:
        json.dump(pos_data, fw, ensure_ascii=False, indent=4)
        
    
    