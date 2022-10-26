bbatch_size=(32 64 96 128 160 192 224 256 288 320 352 384 448 512)
calib_nums=(1024 2048)
calib_path=data/custom_kpts/images
onnx_path=weights/yolov5l6_pose_mix.onnx
trt_path=weights/yolov5l6_pose_mix-INT8.trt
cache_path=caches/yolov5l6_pose_mix.cache
device=1
imgsize=832

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
            --onnx $onnx_path --batch-size 1 --device $device --int8 \
            --calib_path $calib_path \
            --calib_num $num \
            --calib_batch $bs \
            --calib_imgsz $imgsize \
            --cache_dir caches \
            --calib_method MinMax
        python test_multi_backend.py \
            --weights $trt_path \
            --data data/custom_kpts.yaml \
            --img-size $imgsize \
            --conf-thres 0.001 \
            --iou-thres 0.6 \
            --task val \
            --device $device \
            --kpt-label
    done
done

## nohup bash data/scripts/search.sh >> runs/others/search3.txt 2>&1 &