import os
import shutil

# 设置源目录、目标目录基础路径和每个目标目录的文件数量限制
source_dir = '/mnt/pai-storage-ceph-hdd/Dataset/smoke_ty/data_huiliu/head_256/3_smoke_phone'
target_dir_base = '/mnt/pai-storage-ceph-hdd/Dataset/smoke_ty/test'
files_per_folder = 20

def split_into_folders(source, target_base, limit):
    if not os.path.exists(source):
        print(f"Source directory '{source}' not found.")
        return

    batch = 1
    count = 0
    target_dir = f"{target_base}/{batch}"
    os.makedirs(target_dir, exist_ok=True)

    # 使用 os.scandir() 来高效遍历目录
    with os.scandir(source) as it:
        for entry in it:
            if entry.is_file():
                # 这里处理每个文件，例如移动文件
                shutil.move(entry.path, os.path.join(target_dir, entry.name))
                
                count += 1
                if count >= limit:
                    # 当前批次目录已满，准备下一个批次
                    batch += 1
                    count = 0
                    target_dir = f"{target_base}/{batch}"
                    os.makedirs(target_dir, exist_ok=True)

if __name__ == "__main__":
    split_into_folders(source_dir, target_dir_base, files_per_folder)