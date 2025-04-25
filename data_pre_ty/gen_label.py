import glob
import os
import shutil
import random

from tqdm import tqdm
mode = 'train'
# mode = 'val'
# a = ['Celeba','MAFA','shangchao_data']
# a = ['Celeba']
# a = ['coco','CrowdHuman','normal','PANDA']
a = ['normal']

for source_name in a:
    # source_name = 'Celeba'
    print(f'{source_name}数据集处理中...')
    label_dir_1 = fr'/mnt/pai-storage-12/data/phd/{source_name}/{mode}/labels/*'
    label_dir_2 = fr'/mnt/pai-storage-12/data/person_face/{source_name}/{mode}/labels'
    # label_dir_1 = fr'/mnt/pai-storage-14/algorithm/zhouyanggang/datasets/Facedetect_data/yolo-face_data/{source_name}/{mode}/labels/*'
    # label_dir_2 = fr'/mnt/pai-storage-12/data/person_face/{source_name}/{mode}/labels'
    label_dir_3 = fr'/mnt/pai-storage-12/data/person_face/labels_train'
    os.makedirs(label_dir_3,exist_ok=True)
    label_list_1 = glob.glob(label_dir_1)
    image_num = len(label_list_1)
    label_num = 0
    for label_path_1 in tqdm(label_list_1):
        image_name = label_path_1.split('/')[-1]
        label_path_2 = os.path.join(label_dir_2,image_name)
        label_path_3 = os.path.join(label_dir_3,image_name)
        # if os.path.exists(label_path_3):
        #     os.remove(label_path_3)
            # print(f'{image_name}已存在')
            # continue
        data_1 = []
        data_2 = []
        if os.path.exists(label_path_1):
            with open(label_path_1,'r') as f:
                data_1 = [line for line in f.readlines() if not line.startswith('1')]
                # data_1 = f.readlines()
                # data_1 = ['1 ' + line.split(' ', 1)[1] if ' ' in line else '1' for line in data_1]
        if os.path.exists(label_path_2):
            with open(label_path_2,'r') as f:
                data_2 = f.readlines()  
                data_2 = ['1 ' + line.split(' ', 1)[1] if ' ' in line else '1' for line in data_2]
        data_label = data_1 + data_2
        with open(label_path_3,'w') as f:
            for line in data_label:
                f.write(line)
            label_num += 1
        # print(f'{image_name}处理完成')
    
    print(f'{source_name}数据集处理完成,共{image_num}张图片,共{label_num}张标签')
    

