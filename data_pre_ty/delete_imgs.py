import os
import filetype

def check_and_delete_mismatched_files(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 跳过非文件项（如子文件夹）
        if not os.path.isfile(file_path):
            continue
        
        # 检查文件名后缀是否为.jpg
        name, ext = os.path.splitext(filename)
        if ext.lower() != ".jpg":
            continue
        
        # 检测文件真实格式
        kind = filetype.guess(file_path)
        if kind is None:
            continue
        
        # 如果真实格式是 GIF，则删除
        if kind.extension == "gif":
            print(f"删除文件: {filename}(真实格式为 GIF)")
            os.remove(file_path)

if __name__ == "__main__":
    folder = '/mnt/pai-storage-12/data/Facedetect_data/yolo-face_data/PandaEmoji/train/images'
    check_and_delete_mismatched_files(folder)