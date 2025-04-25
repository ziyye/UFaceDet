#!/bin/bash

# 设置 PYTHONPATH 环境变量，将 yolov8 目录添加到 Python 模块搜索路径中
export PYTHONPATH=/mnt/pai-storage-8/tianyuan/face_qrcode_det/yolov8:$PYTHONPATH

# 在后台运行 Python 训练脚本 face_qrcode_train.py
# nohup: 即使关闭终端，进程也会继续运行
# > train.log: 将标准输出重定向到 train.log 文件
# 2>&1: 将标准错误输出重定向到标准输出（即也写入 train.log）
# &: 在后台运行命令
nohup python face_qrcode_train.py > train.log 2>&1 &

echo "训练脚本已在后台启动，日志输出到 train.log" 