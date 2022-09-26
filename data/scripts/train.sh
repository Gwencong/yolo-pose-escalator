# finetune (custom data + coco data)
python train.py \
    --weights weights/yolov5l6_pose.pt \
    --cfg models/hub/yolov5l6_kpts.yaml \
    --data data/merge_kpts.yaml \
    --hyp data/hyp.finetune_evolve.yaml \
    --epochs 200 \
    --batch-size 64 \
    --img-size 832 \
    --device 1,2 \
    --workers 8 \
    --kpt-label \
    --freeze 12 \
    --project runs/finetune
    

# finetune (custom data only)
python train.py \
    --weights weights/yolov5l6_pose.pt \
    --cfg models/hub/yolov5l6_kpts.yaml \
    --data data/custom_kpts.yaml \
    --hyp data/hyp.finetune_evolve.yaml \
    --epochs 200 \
    --batch-size 64 \
    --img-size 832 \
    --device 0 \
    --workers 8 \
    --kpt-label \
    --freeze 12 \
    --project runs/finetune
    

# scratch (custom data only)
python train.py \
    --weights weights/yolov5l6_pose.pt \
    --cfg models/hub/yolov5l6_kpts.yaml \
    --data data/custom_kpts.yaml \
    --hyp data/hyp.scratch.yaml \
    --epochs 300 \
    --batch-size 16 \
    --img-size 832 \
    --device 1,2 \
    --workers 2 \
    --kpt-label \
    --project runs/train
