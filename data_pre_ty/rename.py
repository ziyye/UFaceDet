import os

def rename_images(folder_path):
    # 获取文件夹中所有文件的列表
    files = os.listdir(folder_path)
    
    # 创建一个用于存储图像文件的列表
    images = [file for file in files if file.endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp'))]

    # 对这些图像文件进行排序，以保证按文件名顺序进行重命名
    images.sort()

    # 遍历枚举所有图像，并按递增数字命名
    for index, filename in enumerate(images, start=1):
        # 获取文件的扩展名
        extension = os.path.splitext(filename)[1]
        
        # 新文件名，使用递增的数字
        new_name = f"emoji_{index}{extension}"
        
        # 生成完整的旧文件路径和新文件路径
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_name)
        
        # 使用 os.rename() 方法重命名文件
        os.rename(old_file, new_file)
        print(f"Renamed '{filename}' to '{new_name}'.")

if __name__ == "__main__":
    # 替换为你的图片文件夹路径
    folder_path = "/mnt/pai-storage-8/tianyuan/facedet/data/PandaEmoji"
    
    rename_images(folder_path)