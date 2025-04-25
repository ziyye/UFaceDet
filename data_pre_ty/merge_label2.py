import os
import shutil

def merge_face_labels(face_label_dir, target_dir):
    """
    将人脸标注文件合并到目标目录
    :param face_label_dir: 包含人脸标注文件的目录
    :param target_dir: 目标目录
    """
    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # 遍历人脸标注目录下的所有文件
    for filename in os.listdir(face_label_dir):
        if filename.endswith('.txt'):
            source_path = os.path.join(face_label_dir, filename)
            target_path = os.path.join(target_dir, filename)
            
            # 如果目标文件已存在，则合并内容
            if os.path.exists(target_path):
                with open(source_path, 'r') as src_f, open(target_path, 'r') as tgt_f:
                    src_lines = src_f.readlines()
                    tgt_lines = tgt_f.readlines()
                
                # 合并内容，保留所有标签为0的行
                merged_lines = []
                for line in src_lines:
                    if line.strip().split()[0] == '0':
                        merged_lines.append(line)
                merged_lines.extend(tgt_lines)
                
                # 写回文件
                with open(target_path, 'w') as f:
                    f.writelines(merged_lines)
            else:
                # 如果目标文件不存在，直接复制
                shutil.copy2(source_path, target_path)

if __name__ == '__main__':
    # 指定人脸标注文件目录和目标目录
    face_label_dir = '/mnt/pai-storage-8/tianyuan/facedet/sqyolov8/runs/predict_pt/qrcode2/labels'  # 请替换为实际的人脸标注文件目录
    target_dir = '/data01/data/qrcode/synthetise/val/labels'  # 目标目录
    
    merge_face_labels(face_label_dir, target_dir)