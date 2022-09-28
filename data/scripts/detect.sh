# detect with TensorRT model
python detect_multi_backend.py \
    --weights weights/yolov5l6_pose_custom-FP16.trt \
    --source data/images \
    --device 0 \
    --img-size 832 \
    --kpt-label

# detect with ONNX model
python detect_multi_backend.py \
    --weights weights/yolov5l6_pose_custom.onnx \
    --source data/images \
    --device 0\
    --img-size 832 \
    --kpt-label

# detect with Pytorch model
python detect_multi_backend.py \
    --weights weights/yolov5l6_pose_custom.pt \
    --source data/images \
    --device 0 \
    --img-size 832 \
    --kpt-label
