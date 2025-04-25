import random
from PIL import Image
import glob
import os

from tqdm import tqdm 

# name = 'gyrb'
# new_dir = f'/mnt/pai-storage-ceph-hdd/Dataset/Vehicle/QIDIAN/color_class/images_cls/{name[::-1]}'
new_dir = f'/mnt/pai-storage-ceph-hdd/Dataset/Vehicle/QIDIAN/pos_class/dataset/pos_cls416_0523/3_45trans'

img_dir_list = [
        # f'/mnt/pai-storage-ceph-hdd/Dataset/Vehicle/QIDIAN/color_class/images_cls/{name}/*',
        f'/mnt/pai-storage-ceph-hdd/Dataset/Vehicle/QIDIAN/pos_class/dataset/pos_cls416_0523/3_45/*',
    ]
imgLists = []
for img_dir in img_dir_list:
    imgLists += glob.glob(img_dir)
imgLists = list(filter(lambda path: os.path.splitext(path)[-1].lower() in ['.jpg', '.jpeg', '.png'], imgLists))
random.shuffle(imgLists)

for image_path in tqdm(imgLists):
    new_path1 = os.path.join(new_dir,'rotate180_' + os.path.split(image_path)[-1])
    new_path2 = os.path.join(new_dir,'flippedlr__' + os.path.split(image_path)[-1])
    new_path3 = os.path.join(new_dir,'flippedtb__' + os.path.split(image_path)[-1])

    with Image.open(image_path) as img:
        # 旋转180度
        rotated_img = img.rotate(180)
        flipped_img1 = img.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_img2 = img.transpose(Image.FLIP_TOP_BOTTOM)
        # 展示旋转后的图片
        # rotated_img.show()
        rotated_img.save(new_path1)
        flipped_img1.save(new_path2)
        flipped_img2.save(new_path3)
