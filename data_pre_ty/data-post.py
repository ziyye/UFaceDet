import os
import torch
import torch.nn as nn
import time 
import torchvision
import numpy as np
# import caffe
import onnx 
import onnxruntime
import cv2
from onnx import numpy_helper
from tqdm import tqdm
class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        # print("input_name:{}".format(self.input_name))
        # print("output_name:{}".format(self.output_name))
 
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

grid = [torch.zeros(1)] * 3
anchor_grid = [torch.zeros(1)] * 3
anchors = torch.tensor([[6,   7,  11,  14,   9,  32],  
                        [ 21,  26,  21,  67,  47,  58],  
                        [38, 126,  67, 172, 145, 226]],dtype=torch.float32)
stride = torch.tensor([8,16,32],dtype=torch.float32)
colors = [(56, 56, 255),(151, 157, 255),(31, 112, 255)]

def pre_process(img_path,size):  
    img_mat = cv2.imread(img_path)
    im0 , r, (dw,dh) = letterbox(img_mat,size)
    im = im0.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)
    im = im[np.newaxis, :].astype(np.float32)
    im /= 255
    # im = torch.from_numpy(im,dtype = torch.float)
    return im,im0

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




def make_grid(nx=20, ny=20, i=0):
        shape = 1, 3, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device='cpu', dtype=torch.float32), torch.arange(nx, device='cpu', dtype=torch.float32)
        yv, xv = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (anchors[i]).view((1, 3, 1, 1, 2)).expand(shape)
        return grid, anchor_grid

def post_process(x):
    z = []  # inference output
    for i in range(3):
        bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        y = x[i].view(bs, 3, 7, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        # x[i] = x[i].view(bs, 3, 9, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        grid[i], anchor_grid[i] = make_grid(nx, ny, i)
        # y = x[i].sigmoid()      
        y[..., 0:2] = (y[..., 0:2] * 2 + grid[i]) * stride[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) *(y[..., 2:4] * 2) * anchor_grid[i]  # wh
        z.append(y.view(bs, -1, 7))
    return torch.cat(z, 1)

def non_max_suppression(prediction,
                        conf_thres=0.6,
                        iou_thres=0.45,
                        classes=None,
                        agnostic=False,
                        multi_label=False,
                        labels=(),
                        max_det=300):
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence

        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]
    return output
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
def box_iou(box1, box2, eps=1e-7):
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter + eps)
def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])

def plot_img(img_path,preds,save_path):
    img0 = cv2.imread(img_path)
    for pred in preds:
        box = [int(a) for a in pred[:4]]
        conf = str(float(pred[-2]))[:4]    
        cls = int(pred[-1])
        cv2.rectangle(img0, (box[0], box[1]),(box[2], box[3]), colors[cls], 3,lineType=cv2.LINE_AA)
        cv2.putText(img0, str(cls) + ' ' + str(conf), (box[0], box[1]-5), 0, 0.6, (255, 255, 255), 2, lineType=cv2.LINE_AA)      
    cv2.imwrite(save_path,img0)
    return

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
        txt_path = fr'/mnt/pai-storage-1/tianyuan/workspace/smoke/sqyolov8/select/time_20240620_2501.txt'

        # 目标文件夹的路径
        select_path = fr'/mnt/pai-storage-ceph-hdd/Dataset/smoke_ty/data_huiliu/202312after'

        # 确保目标目录存在
        os.makedirs(select_path, exist_ok=True)

        with open(txt_path, 'r') as f:
            data = [x.strip().replace('/home/shuiwei/sw_79','/mnt/pai-storage-2') for x in f.read().strip().splitlines() if len(x)]

        # 检查每一个入口
        for img_path in tqdm(data):
            
            # 确定目标路径
            # img_path = img_p.replace('/home/shuiwei/sw_79','/mnt/pai-storage-2')
            target_path = os.path.join(select_path, os.path.split(img_path)[-1])
            orig_img = cv2.imread(img_path)
            # 前处理
            im,im0 = pre_process(img_path,416)
            # im_tensor = torch.tensor(im,dtype=torch.float)

            # ONNX 模型载入推理
            onnx_path = r"/mnt/pai-storage-1/tianyuan/workspace/smoke/sqyolov8/select/PHD_V3.1.1.20240417_bgr_yolov7_prunes.onnx"
            onnx_test = ONNXModel(onnx_path)
            onnx_result = onnx_test.forward(im)
            outputs = [torch.tensor(onnx_result[i],dtype=torch.float32) for i in range(3)]

            # NMS
            preds = post_process(outputs)
            pred = non_max_suppression(preds,conf_thres=0.1)[0]
            pred[:, :4] = scale_boxes(im.shape[2:], pred[:, :4], orig_img.shape)

            for j,pre in enumerate(pred):
                box = [int(a) for a in pre[:4]]
                w = box[2]-box[0]
                h = box[3]-box[1]
                x1 = int(box[0] - w/2)
                x2 = int(box[2] + w/2)
                y1 = int(box[1] - h/2)
                y2 = int(box[3] + h/2)
                conf = str(float(pre[-2]))[:4]    
                cls = int(pre[-1])

                if cls == 1 :
                    save_path = target_path[:-4] + fr'_{j+1}.jpg'
                    x1 = 0 if x1<0 else x1 
                    y1 = 0 if y1<0 else y1 
                    x2 = orig_img.shape[1] if x2>orig_img.shape[1] else x2
                    y2 = orig_img.shape[0] if y2>orig_img.shape[0] else y2
                    head = orig_img[y1:y2,x1:x2,:]
                    # head1 = letterbox(head,256)[0]
                    cv2.imwrite(save_path,head)
                    print(f'{save_path} is saved!')
