# export onnx
python models/export_onnx.py \
    --weights weights/yolov5l6_pose_custom.pt \
    --img-size 832 \
    --device 0 \
    --batch-size 1 \
    --simplify \
    --half

# export onnx end2end
python models/export_onnx.py \
    --weights weights/yolov5l6_pose_custom.pt \
    --img-size 832 \
    --batch-size 1 \
    --device 0 \
    --simplify \
    --half \
    --end2end \
    --topk-all 100 \
    --iou-thres 0.45 \
    --conf-thres 0.5 

# export TensorRT with scripts
python models/export_TRT.py \
    --onnx weights/yolov5l6_pose_custom-NMS.onnx \
    --batch-size 1 \
    --device 1 \
    --fp16

# export TensorRT with trtexec
trtexec \
    --onnx=weights/yolov5l6_pose.onnx \
    --workspace=4096 \
    --saveEngine=weights/yolov5l6_pose.trt \
    --fp16


# export TensorRT with int8
python models/export_TRT.py \
    --onnx weights/yolov5l6_pose_custom-FP16.onnx \
    --batch-size 1 \
    --device 1 \
    --int8 \
    --calib_path data/custom_kpts/images \
    --calib_num 1024 \
    --calib_batch 128 \
    --calib_imgsz 832 \
    --cache_dir caches \
    --calib_method Entropy \
    --calib_letterbox