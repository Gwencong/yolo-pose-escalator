batch_size=(32 64 128 192 256 320 384 448 512)
calib_nums=(1024 2048)
calib_path=data/custom_kpts/images
onnx_path=weights/yolov5l6_pose_custom-FP16.onnx
trt_path=weights/yolov5l6_pose_custom-FP16-INT8.trt
cache_path=caches/yolov5l6_pose_custom-FP16.cache

echo "Current path: $PWD"
for bs in "${batch_size[@]}"  
do  
    for num in "${calib_nums[@]}" 
    do
        echo " "
        echo "Calib batch size: ${bs}, calib nums: ${num}";
        rm -rf $cache_path
        echo "remove cache file $cache_path"
        python models/export_TRT.py \
            --onnx $onnx_path --batch-size 1 --device 1 --int8 \
            --calib_path $calib_path \
            --calib_num $num \
            --calib_batch $bs \
            --calib_imgsz 832 \
            --cache_dir caches \
            --calib_method MinMax
        python test_multi_backend.py \
            --weights $trt_path \
            --data data/custom_kpts.yaml \
            --img-size 832 \
            --conf-thres 0.001 \
            --iou-thres 0.6 \
            --task val \
            --device 1 \
            --kpt-label
    done
done

## nohup bash data/scripts/search.sh >> runs/others/search.txt 2>&1 &