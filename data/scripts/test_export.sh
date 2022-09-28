# test TensorRT model
python test_multi_backend.py \
    --weights weights/yolov5l6_pose_custom-FP16-INT8.trt \
    --data data/custom_kpts.yaml \
    --img-size 832 \
    --conf-thres 0.001 \
    --iou-thres 0.6 \
    --task val \
    --device 0 \
    --kpt-label

# test ONNX model
python test_multi_backend.py \
    --weights weights/yolov5l6_pose_custom.onnx \
    --data data/custom_kpts.yaml \
    --img-size 832 \
    --conf-thres 0.001 \
    --iou-thres 0.6 \
    --task val \
    --device 0 \
    --kpt-label

# test Pytorch model
python test_multi_backend.py \
    --weights weights/yolov5l6_pose_custom.pt \
    --data data/custom_kpts.yaml \
    --img-size 832 \
    --conf-thres 0.001 \
    --iou-thres 0.6 \
    --task val \
    --device 0 \
    --kpt-label