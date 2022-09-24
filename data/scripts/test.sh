# test coco data
python test.py \
    --weights runs/fintune/exp/weights/best.pt \
    --data data/coco_kpts.yaml \
    --task val \
    --batch-size 32 \
    --img-size 832 \
    --device 1 \
    --kpt-label

# test custom data
python test.py \
    --weights runs/fintune/exp/weights/best.pt \
    --data data/custom_kpts.yaml \
    --task test \
    --batch-size 32 \
    --img-size 832 \
    --device 1 \
    --kpt-label