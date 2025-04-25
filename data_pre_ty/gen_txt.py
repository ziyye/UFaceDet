import glob
import os
import shutil
import random
# mode = 'train'
mode = 'val'
# a = ['Celeba','MAFA','shangchao_data','Uface','Uface_register','widerface','background','hand','animals','PandaEmoji'] # train
# a = ['Celeba','MAFA','shangchao_data'] # val
# a = ['public_1','public_2','synthetise'] # code train
a = ['public_1','synthetise'] # code val
# a = ['selfmake_dataset','warning_dataset_1','warning_dataset_2']
# a = ['hand']
imgList = []
txt_path = fr'/mnt/pai-storage-8/tianyuan/face_qrcode_det/yolov8/datasets/face_qrcode_{mode}.txt'
for dir in a:
    print(f'{dir}数据集加载中...')
    # images_dir = fr"/data/data/Facedetect_data/yolo-face_data/{dir}/{mode}/images/*"
    images_dir = fr"/data/data/qrcode_data/qrcode/{dir}/{mode}/images/*"
    data = glob.glob(images_dir)
    random.shuffle(data)
    if dir == 'Celeba':
        dataset = data[:10000]
        # dataset = data[:600]
    elif dir == 'MAFA':
        dataset = data[:10000]
        # dataset = data[:600]
    elif dir =='shangchao_data':
        dataset = data[:20000]
        # dataset = data[:600]
    elif dir == 'Uface':
        dataset = data[:50000]
        # dataset = data[:1200]
    elif dir == 'Uface_register':
        dataset = data[:10000]
        # dataset = data[:200]
    elif dir == 'widerface':
        dataset = data*8
        # dataset = data[:]
    elif dir == 'hand':
        dataset = data*3
    elif dir == 'background':
        dataset = data
    elif dir == 'animals':
        dataset = data*2
    elif dir == 'PandaEmoji':
        dataset = data*2
    elif dir == 'public_1':
        dataset = data*2
    elif dir == 'public_2':
        dataset = data*2
    elif dir == 'synthetise':
        dataset = data[:2000]
    else:
        dataset = data
    imgList += dataset
    print(f'{dir}分配{len(dataset)}张图片,总数据集共{len(imgList)}张图片')
random.shuffle(imgList)
with open(txt_path,'a') as f:
    for img_path in imgList:
        f.write(img_path + '\n')
with open(txt_path,'r') as f:
    lenth = len(f.readlines())
print(f'{mode}数据集共{lenth}张图片')

