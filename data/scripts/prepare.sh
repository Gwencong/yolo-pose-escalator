# 准备 YOLO-POSE 格式的 label 文件
python escalator/json2yolo.py \
    --json_dir data/custom_kpts/annotations \
    --label_dir data/custom_kpts/labels

# 划分数据集
python escalator/split.py \
    --image_dir data/custom_kpts/images \
    --prefix_path ./images/ \
    --out_path data/custom_kpts \
    --split 0.7 0.2 0.1

# 抽取等量coco数据
python escalator/extract_equivalent_coco.py \
    --custom_data data/custom_kpts/train.txt \
    --coco_data data/coco_kpts/train2017.txt 

# 随机抽取可视化
python escalator/visual.py \
    --data_root data/custom_kpts \
    --data_file train.txt \
    --visual_num 5 \
    --save_dir runs/visual