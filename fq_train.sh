#!/bin/bash
export MKL_THREADING_LAYER=GNU
# 设置 PYTHONPATH 环境变量，将 yolov8 目录添加到 Python 模块搜索路径中
export PYTHONPATH=/mnt/pai-storage-8/tianyuan/face_qrcode_det/yolov8:$PYTHONPATH

# 定义日志目录
LOG_DIR="/mnt/pai-storage-8/tianyuan/face_qrcode_det/yolov8/logs"
# 获取当前日期 (格式: YYYY-MM-DD)
CURRENT_DATE=$(date +%Y-%m-%d)
# 构造完整的日志文件路径
LOG_FILE="$LOG_DIR/train_$CURRENT_DATE.log"

# 检查日志目录是否存在，如果不存在则创建
mkdir -p "$LOG_DIR"

# 在后台运行 Python 训练脚本 face_qrcode_train.py
# nohup: 即使关闭终端，进程也会继续运行
# > "$LOG_FILE": 将标准输出重定向到指定的日期日志文件
# 2>&1: 将标准错误输出重定向到标准输出（即也写入日志文件）
# &: 在后台运行命令
nohup python face_qrcode_train.py > "$LOG_FILE" 2>&1 &

echo "训练脚本已在后台启动，日志输出到 $LOG_FILE" 