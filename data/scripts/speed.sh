trtexec --avgRuns=100 --loadEngine=weights/yolov5l6_pose_custom-cpu-INT8.trt
trtexec --avgRuns=100 --loadEngine=weights/yolov5l6_pose_custom-cpu-FP16.trt
trtexec --avgRuns=100 --loadEngine=weights/yolov5l6_pose_custom-INT8.trt
trtexec --avgRuns=100 --loadEngine=weights/yolov5l6_pose_custom-FP16.trt