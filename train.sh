# conda activate yolo
# export HOME=/data/wangjiazhi

export PYTHONPATH=./:$PYTHONPATH
# nohup python face_qrcode_train.py > train.log 2>&1 &
export TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# export TIMESTAMP="Debug"
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# export CUDA_VISIBLE_DEVICES="1,2,3,4,5,6"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
# export CUDA_VISIBLE_DEVICES="0,1"
python face_qrcode_train.py 2>&1 | tee logs/train_${TIMESTAMP}.log


# PYTHONPATH=/data/wangjiazhi/projs/yolov8_0512:$PYTHONPATH CUDA_VISIBLE_DEVICES="0,1,2,3" python face_qrcode_train-debug.py
