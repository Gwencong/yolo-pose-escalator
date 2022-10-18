# YOLO-Pose TensorRT部署
此仓库提供了将YOLO-Pose利用TensorRT加速的相关脚本，在YOLO-Pose源代码基础上进行了添加和修改，主要内容如下：
1. 导出onnx模型，包括FP16精度导出、带NMS的模型导出
2. 导出TensorRT模型，包括FP16和INT8
3. 测试和推理导出的模型，包括pytorch、onnx、TensorRT模型
4. 处理和训练自定义数据
5. 修改了forward函数，部署TensorRT后比优化前速度更快，修改内容见[这里](docs/add_no_inplace)

## 1.模型导出
### （1）导出onnx
onnx的导出脚本为`models/export_onnx.py`，支持
* 导出一般的动态、静态onnx模型
* 导出带nms的模型(--end2end)
* 导出FP16的onnx模型（--half）

导出带NMS的onnx模型的目的是为了导出带NMS的TensorRT模型，即把NMS这一过程集成到模型推理中。由于YOLO-Pose的输出除了box还有pose，因此插入的NMS算子要能够输出经过NMS后的box索引，以获取box对应的pose，满足此条件的TensorRT官方plugin只有EfficientNMS-ONNX，onnx和pytorch中并没有该算子，因此导出的含NMS的onnx模型只有转化为TensorRT模型后才可进行推理。导出实例如下：
```bash
# 导出onnx模型
python models/export_onnx.py \
	--weights weights/yolov5l6_pose.pt \
	--img-size 832 \
	--device 0 \
	--batch-size 1 \
	--simplify \
	--half

# 导出带onnx的模型
python models/export_onnx.py \
	--weights weights/yolov5l6_pose.pt \
	--img-size 832 \
	--batch-size 1 \
	--device 0 \
	--simplify \
	--half \
	--end2end \
	--topk-all 100 \
	--iou-thres 0.45 \
	--conf-thres 0.5

'''
参数含义：更多释义详见代码
python models/export_onnx.py \
  --weights     pytorch模型路径
  --img-size    导出onnx模型的输入尺寸
  --batch-size  导出onnx模型的batch size
  --device      在cpu上导出还是在gpu上导出
  --dynamic     导出动态onnx模型
  --half        导出FP16权重的onnx模型,device需为gpu
  --simplify    使用onnx-simplifier简化模型
  --end2end     导出含NMS的模型
  --topk-all    NMS保留的最大box/pose数量
  --iou-thres   NMS的iou阈值
  --conf-thres  NMS的置信度阈值
'''
```

### （2）导出TensorRT模型
导出脚本为`models/export_onnx.py`，支持导出FP16模型和INT8量化模型，也可以直接使用trtexec导出TensorRT模型
```bash
# export TensorRT with scripts
python models/export_TRT.py \
	--onnx weights/yolov5l6_pose.onnx \
	--batch-size 1 \
	--fp16
	
# export TensorRT with trtexec
trtexec \
	--onnx=weights/yolov5l6_pose.onnx \
	--workspace=4096 \
	--saveEngine=weights/yolov5l6_pose.trt \
	--fp16

# export TensorRT with INT8 precision
python models/export_TRT.py \
	--onnx weights/yolov5l6_pose.onnx \
	--batch-size 1 \
	--device 0 \
	--int8 \
	--calib_path data/custom_kpts/images \
	--calib_num 1024 \
	--calib_batch 128 \
	--calib_imgsz 832 \
	--cache_dir caches
	
'''
参数含义: 更多释义详见代码
python models/export_TRT.py 
	--onnx				onnx模型路径
	--batch-size		导出TensorRT模型的batch size，只有onnx为动态模型时有效
	--device			导出TensorRT时使用的GPU id
	--fp16				导出FP16精度的TensorRT模型
	--int8				导出INT8精度的TensorRT模型
	--workspace			导出TensorRT时最多可以使用的GPU显存
	--verbose			输出详细信息
	--dynamic			导出动态TensorRT模型，只有在onnx模型是动态时有效
	--calib_path		校准图片所在路径
	--calib_num			校准图片使用数量
	--calib_batch		校准图片batch size
	--calib_imgsz		校准图片尺寸
	--calib_method		校准方式，提供MinMax和Entropy两种
	--calib_letterbox	是否进行灰边填充
	--cache_dir			校准缓存文件保持文件夹
'''
```

## 2.推理和测试导出模型精度
代码中提供了pytorch、onnx、TensorRT模型的推理脚本和测试导出模型精度的脚本
### （1）推理
推理脚本为`detect_multi_backend.py`，同时支持pytorch、onnx、TensorRT模型的推理，命令如下
```bash
# detect with TensorRT model
python detect_multi_backend.py \
    --weights weights/yolov5l6_pose-FP16.trt \
    --source data/images \
    --device 0 \
    --img-size 832 \
    --kpt-label

# detect with ONNX model
python detect_multi_backend.py \
    --weights weights/yolov5l6_pose.onnx \
    --source data/images \
    --device 0\
    --img-size 832 \
    --kpt-label

# detect with Pytorch model
python detect_multi_backend.py \
    --weights weights/yolov5l6_pose.pt \
    --source data/images \
    --device 0 \
    --img-size 832 \
    --kpt-label

## 参数比较简单，和detect.py一样，详见代码
```
### （2）测试
测试脚本为`test_multi_backend.py`，同时支持pytorch、onnx、TensorRT模型的测试，测试前需要按照[准备数据](#prepare)中的步骤准备好测试集，并设置好数据配置文件的测试集路径，命令如下
```bash
# test TensorRT model
python test_multi_backend.py \
    --weights weights/yolov5l6_pose-INT8.trt \
    --data data/coco_kpts.yaml \
    --img-size 832 \
    --conf-thres 0.001 \
    --iou-thres 0.6 \
    --task val \
    --device 0 \
    --kpt-label

# test ONNX model
python test_multi_backend.py \
    --weights weights/yolov5l6_pose.onnx \
    --data data/coco_kpts.yaml \
    --img-size 832 \
    --conf-thres 0.001 \
    --iou-thres 0.6 \
    --task val \
    --device 0 \
    --kpt-label

# test Pytorch model
python test_multi_backend.py \
    --weights weights/yolov5l6_pose.pt \
    --data data/coco_kpts.yaml \
    --img-size 832 \
    --conf-thres 0.001 \
    --iou-thres 0.6 \
    --task val \
    --device 0 \
    --kpt-label

## 参数比较简单，和test.py一样，详见代码
```

## 3. 算法训练
### (1) 获取训练代码
运行如下命令克隆训练代码，并安装训练代码的环境依赖：
```bash
# 1.获取训练代码（yolo-pose源码+训练脚本和数据处理脚本以及相关优化）
git clone --recurse https://github.com/Gwencong/yolo-pose-escalator.git

# 2.安装环境依赖
cd yolo-pose-escalator & pip install -r requirements.txt
```
### (2) <a id="prepare">准备数据</a>
按照如下步骤进行：
1. 将COCO数据集或自定义数据集组织为如下格式，其中自定义数据的json标注文件为每张图片对应的标注文件，格式和coco类似，包含图片中所有人的关键点和边界框的绝对坐标，如果事先已经转换为YOLO-Pose格式的txt标注文件，可以跳过这一步
```bash
...
data
  |--coco_kpts
  |		|--images
  |		 	|--train2017
  |		 		|--image1.jpg
  |		 		|--image2.jpg
  |		 		|--...
  |		 	|--val2017
  |		 		|--image3.jpg
  |		 		|--image4.jpg
  |		 		|--...
  |		|--labels  # https://drive.google.com/file/d/1irycJwXYXmpIUlBt88BZc2YzH__Ukj6A/view 下载
  |		 	|--train2017
  |		 		|--image1.txt
  |		 		|--image2.txt
  |		 		|--...
  |		 	|--val2017
  |		 		|--image3.txt
  |		 		|--image4.txt
  |		 		|--...
  |		|--train2017.txt
  |		|--tval2017.txt
  |
  |--custom_kpts
  |		|--images
  |		 	|--image1.jpg
  |		 	|--image1.jpg
  |		|--annotations
  |		 	|--image1.json
  |		 	|--image1.json
  |--...
...
```
2. 将自定义数据集的json标注文件转为YOLO格式的txt标注文件（如果使用COCO数据集可以跳过步骤234）
```bash
# 运行如下命令，将在data/custom_kpts下生成labels文件夹，里面保存了转换的txt标注文件
python escalator/json2yolo.py \
	--json_dir data/custom_kpts/annotations \
	--label_dir data/custom_kpts/labels

## 参数含义
## --json_dir	json标注文件所在文件夹
## --label_dir	txt标注文件保存的文件夹
```
3. 划分数据集
```bash
# 运行如下命令，划分训练集、验证集、测试集，将在data/custom_kpts生成train.txt，val.txt，test.txt
python escalator/split.py \
	--image_dir data/custom_kpts/images \
	--prefix_path ./images/ \
	--out_path data/custom_kpts \
	--split 0.7 0.2 0.1
	
## 参数含义
## --image_dir: 	图片路径
## --prefix_path	train.txt中文件路径前缀
## --out_path		train.txt文件保存的文件夹
## --split			训练集：验证集：测试集的划分比例
```

4.  随机可视化数据，确保label转换正常
```bash
# 随机抽取转换的txt文件可视化，可视化结果保存在runs/visual文件夹中
python escalator/visual.py \
	--data_root data/custom_kpts \
	--data_file train.txt \
	--visual_num 5 \
	--save_dir runs/visual

## 参数含义
## --data_root 	数据集目录
## --data_file 	训练集文件
## --visual_num 可视化数量
## --save_dir 	可视化结果保存路径
```
### (3) 训练
1. 从头训练（scratch）
```bash
# scratch (custom data only)
python train.py \
	--weights weights/yolov5l6_pose.pt \
	--cfg models/hub/yolov5l6_kpts.yaml \
	--data data/custom_kpts.yaml \
	--hyp data/hyp.scratch.yaml \
	--epochs 300 \
	--batch-size 16 \
	--img-size 832 \
	--device 0,1 \
	--workers 8 \
	--kpt-label \
	--project runs/train

## 参数含义: 详见官方文档以及代码
```
2. 微调（finetune）
```bash
# finetune (custom data only)
python train.py \
	--weights weights/yolov5l6_pose.pt \
	--cfg models/hub/yolov5l6_kpts.yaml \
	--data data/custom_kpts.yaml \
	--hyp data/hyp.finetune_evolve.yaml \
	--epochs 200 \
	--batch-size 64 \
	--img-size 832 \
	--device 0,1 \
	--workers 8 \
	--kpt-label \
	--freeze 12 \
	--project runs/finetune
## 参数含义: 详见官方文档以及代码
```
## 4. 测试结果
在自定义数据集上训练后，导出训练后的yolov5l6模型，速度测试的环境为Jetson Xavier NX，测试工具为trtexec，测试结果如下：
<p align="center"><font face="黑体" size=3.>表1 自定义数据集测试结果</font></p>
<div align="center">

|     精度      |   测试尺寸   |     mAP     |  速度(Jetson Xavier NX)  |
|    :----:    | :---------: |  :-------:  | :----: |
| Pytorch FP32 |  832×832    |    0.803    |   -    |
| Pytorch FP16 |  832×832    |    0.803    |   -    |
| ONNX FP16    |  832×832    |    0.803    |   -    |
| TensorRT FP16|  832×832    |    0.803    | 70.80 ms |
| TensorRT INT8|  832×832    |    0.789    | 42.90 ms |

</div>

<p align="center"><font face="黑体" size=3.>表2  模型NMS测试结果</font></p>
<div align="center">

| NMS  |precision|   测试尺寸   | conf thresh | iou thresh |    mAP  | speed in NX (ms) |
|:----:|  :----: | :---------: |  :-------:  |  :------:  | :-----: |:---------------: |
|      |   FP16  |  832×832    |    0.45     |     0.5    |  0.776  |     70.80     |
|   √  |   FP16  |  832×832    |    0.45     |     0.5    |  0.769  |     71.16     |
|      |   INT8  |  832×832    |    0.45     |     0.5    |  0.762  |     42.90     |
|   √  |   INT8  |  832×832    |    0.45     |     0.5    |  0.750  |     43.76     |

</div>

&emsp;&emsp;上表同时列出了INT8与FP16的测试结果，由于训练时采用混合精度训练，所以从上表3的结果可以看出，FP16和FP32的精度是一致的，从pytoch到onnx再到TensorRT的过程中并无精度损失。从表4来看，模型加入NMS后，推理耗时增加了1ms左右，说明将NMS放与模型内，其耗时在1ms左右，而如果放在模型外，经测试，使用pytorch自带的NMS，耗时在2ms以上，需要说明的是，NMS本身耗时不多，将NMS放在模型内的主要目的一是端到端，不用在后处理部分进行NMS，二是在DeepStream中，如果提前在模型内做完NMS，那么DeepStream输出的Tensor会大大减小，经过测试对于DeepStream有一定的速度提升。对于检测精度的影响而言，NMS放入模型后，从表2的mAP测试结果可以看出，精度有所下降，但在可接受范围内，FP16的情况下下降0.7个点，对检测效果基本无影响。

&emsp;&emsp;从表1结果可以看出，在Jetson Xavier NX上，INT8与FP16相比模型mAP下降1.4左右，对于检测性能影响较小，而推理耗时可减少28ms，取得更快的推理速度。
&emsp;&emsp;在Jetson Xavier NX上进行INT8量化时，可以将量化数据（采集的数据，1024张即可）拷贝到NX设备上，然后进行量化，也可将在服务器上转换完成后生成的int8量化缓存文件（caches文件夹下的.cache文件）和onnx模型文件放在NX上进行转化而无需拷贝数据到NX上。

## 5. 其他注意事项
1. NMS模型使用了EfficientNMS算法，对于iou阈值和置信度阈值较为敏感，阈值较低会导致模型无法输出正确结果，建议导出onnx模型时iou阈值大于0.45，置信度阈值大于0.5
2. 代码中的INT8量化使用的是PTQ量化，QAT量化有尝试过但是失败了（导出到onnx模型推理结果都正常，但是导出TensorRT后无法得到正确结果不知道为啥。。。）
3. INT8 PTQ量化有两种常用的校准器：最大最小校准器（MinMaxCalibrator）和熵校准器（EntropyCalibrator2），实际测试对于YOLO-Pose最大最小校准器掉点更少
4. INT8 PTQ量化时，发现如果在校准数据预处理时开启灰边填充（letterbox）并且设置参数`auto=True`时会掉点较多，实际发现是在训练虽然有开启灰边填充和mosaic，但是参数`auto=False`，引入的灰边较少；而如果设置参数`auto=True`会引入较多灰边，影响校准时的数据直方图，实际测试关闭灰边填充掉点最少
5. 写TensorRT推理代码时，如果使用pytorch代替pycuda进行数据拷贝，需要注意模型的输入图片数据精度需要与输入层的数据精度一样，否者无法得到正确的推理结果
6. pytorch和pycud同时使用时可能产生一些奇怪的错误，可能与cuda资源的初始化有关

## Reference
[1] [Official YOLOV5 repository](https://github.com/ultralytics/yolov5/)  
[2] [Official YOLO-Pose repository](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)