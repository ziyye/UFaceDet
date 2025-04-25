import os
import shutil
import cv2
import numpy as np

def ellipse_to_yolo(ellipse_data, img_width, img_height):
    """
    将椭圆标注信息转换为YOLO格式。
    ellipse_data: 椭圆信息的元组，格式为(主轴半径, 副轴半径, 旋转角度, 中心点x坐标, 中心点y坐标)
    img_width: 图像的宽度
    img_height: 图像的高度
    
    返回YOLO格式的标注：(类别ID, 中心点x坐标, 中心点y坐标, 宽度, 高度)，
    其中坐标和尺寸都被归一化为[0, 1]区间内的值。
    """
    major_axis_radius, minor_axis_radius, rotation, center_x, center_y, _ = ellipse_data
    
    # 假设椭圆近似为其最小外接矩形，忽略旋转角度
    bbox_width = 2 * major_axis_radius
    bbox_height = 2 * minor_axis_radius
    
    # 归一化坐标和尺寸
    x_center_norm = center_x / img_width
    y_center_norm = center_y / img_height
    width_norm = bbox_width / img_width
    height_norm = bbox_height / img_height
    
    # 类别ID，假定是0
    class_id = 0
    
    return (class_id, x_center_norm, y_center_norm, width_norm, height_norm)
def fddb_to_yolo(ellipse_params, img_size):
    """
    Convert FDDB ellipse format to YOLO bounding box format.
    
    Parameters:
    - ellipse_params: (major_axis_radius, minor_axis_radius, angle, center_x, center_y)
      for the ellipse in the FDDB dataset.
    - img_size: (width, height) of the image.
    
    Returns:
    A string representing the bounding box in YOLO format: "class cx cy w h"
    where `class` is always 0 for faces, (cx, cy) are the center of the box relative to the width and height,
    and (w, h) are the width and height of the box relative to the image size.
    """
    a, b, angle, center_x, center_y,_ = ellipse_params
    img_width, img_height = img_size

    # Convert angle from radians to degrees
    angle_degrees = np.degrees(angle)

    # Generate points on the ellipse
    theta = np.linspace(0, 2*np.pi, 1000)
    x_ellipse = center_x + a * np.cos(theta) * np.cos(angle) - b * np.sin(theta) * np.sin(angle)
    y_ellipse = center_y + a * np.cos(theta) * np.sin(angle) + b * np.sin(theta) * np.cos(angle)

    # Find min and max of the points
    xmin = min(x_ellipse) if min(x_ellipse)>0 else 0
    xmax = max(x_ellipse) if max(x_ellipse)<img_width else img_width
    ymin = min(y_ellipse) if min(y_ellipse)>0 else 0
    ymax = max(y_ellipse) if max(y_ellipse)<img_height else img_height

    # xmin, xmax = min(x_ellipse), max(x_ellipse)
    # ymin, ymax = min(y_ellipse), max(y_ellipse)

    # Compute bounding box in image coordinates
    bbox_width = xmax - xmin
    bbox_height = ymax - ymin
    bbox_center_x = xmin + bbox_width / 2.0
    bbox_center_y = ymin + bbox_height / 2.0

    # Normalize for YOLO
    cx = bbox_center_x / img_width
    cy = bbox_center_y / img_height
    w = bbox_width / img_width
    h = bbox_height / img_height

    # Assuming class 0 for faces
    return "0 {:.6f} {:.6f} {:.6f} {:.6f}".format(cx, cy, w, h)

def load_fddb_annotation(file_path,img_dir,new_imgs_dir,new_labels_dir):

    """
    加载FDDB椭圆标注数据。假设file_path是FDDB数据集标注文件的路径。
    返回一个列表，其中每个元素是一个字典，包含了'path'和'faces'两个键。
    """
    data_list = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            record = {}
            path = lines[i].strip()
            name = path.replace('/','_')
            img_path = img_dir + path + '.jpg'
            new_img_path = new_imgs_dir + name + '.jpg'
            new_label_path = new_labels_dir + name + '.txt'
            shutil.copy(img_path, new_img_path)
            img = cv2.imread(new_img_path)
            h, w = img.shape[:2]
            i += 1
            num_faces = int(lines[i].strip())
            i += 1
            faces = []
            for _ in range(num_faces):
                face_data = tuple([float(num) for num in lines[i].strip().split()])
                face_data = fddb_to_yolo(face_data, (w, h))
                faces.append(face_data)
                i += 1
            with open(new_label_path, 'w') as f:
                for face in faces:
                    f.write(face + '\n')

                    # f.write(f"{face[0]} {face[1]} {face[2]} {face[3]} {face[4]}\n")
    return data_list
def image_copy(file_path,img_dir,new_imgs_dir):

    data_list = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            path = lines[i].strip()
            name = path.replace('/','_')

            img_path = img_dir + path + '.jpg'
            new_img_path = new_imgs_dir + name + '.jpg'
            if os.path.exists(new_img_path):
                print(img_path)
            shutil.copy(img_path, new_img_path)
            i += 1
if __name__ == '__main__':
    # for i in range(1,10):
        annotation_file = f'/mnt/pai-storage-ceph-hdd-nfs/Dataset/FDDB/FDDB-folds/FDDB-fold-10-ellipseList.txt'
        # annotation_file = f'/mnt/pai-storage-ceph-hdd-nfs/Dataset/FDDB/FDDB-folds/FDDB-fold-0{i}-ellipseList.txt'
        images_dir = '/mnt/pai-storage-ceph-hdd-nfs/Dataset/FDDB/'
        new_imgs_dir = '/mnt/pai-storage-ceph-hdd-nfs/Dataset/FDDB/images/'
        labels_dir = '/mnt/pai-storage-ceph-hdd-nfs/Dataset/FDDB/labels/'
        data = load_fddb_annotation(annotation_file,images_dir,new_imgs_dir,labels_dir)

        