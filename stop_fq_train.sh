#!/bin/bash

# 定义要查找的模式，基于实际运行的进程命令行特征
PATTERN="Ultralytics/DDP"
# 或者，如果你想更精确，可以用临时脚本的路径，但这可能每次都变
# PATTERN="/home/tianyuan/.config/Ultralytics/DDP/_temp_"

echo "尝试终止包含 \"$PATTERN\" 的训练进程..."

# 使用 pkill -f 查找并强制终止包含指定模式的进程
# -f: 匹配完整命令行参数
# -9: 发送 SIGKILL 信号
pkill -9 -f "$PATTERN"

# 检查 pkill 命令的退出状态
EXIT_STATUS=$?

if [ $EXIT_STATUS -eq 0 ]; then
  echo "已成功发送终止信号给匹配 '$PATTERN' 的进程。"
elif [ $EXIT_STATUS -eq 1 ]; then
  echo "未找到正在运行的、匹配 '$PATTERN' 的训练进程。"
else
  echo "终止进程时发生错误 (退出状态: $EXIT_STATUS)。请检查权限。"
fi

echo "中断脚本执行完毕。"
