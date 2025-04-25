import cv2
import os

def extract_frames(video_path, output_folder, frame_interval=15):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 每15帧保存一张图像
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"总共保存了 {saved_count} 张图像到 {output_folder}")

# 使用示例
video_path = '/mnt/pai-storage-1/tianyuan/workspace/facedet/sqyolov8/test_imgs/AIPCAM_1226_1_3.mp4'
output_folder = '/mnt/pai-storage-1/tianyuan/workspace/facedet/sqyolov8/test_imgs/AIPCAM_1226_1_3_imgs'
extract_frames(video_path, output_folder)