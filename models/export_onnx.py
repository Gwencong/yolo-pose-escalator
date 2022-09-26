"""Exports a YOLOv5 *.pt model to ONNX and TorchScript formats

Usage:
    $ export PYTHONPATH="$PWD" && python models/export.py --weights yolov5s.pt --img 640 --batch 1
"""

import sys
import time
import warnings
import argparse
import traceback
from pathlib import Path


sys.path.append(Path(__file__).parent.parent.absolute().__str__())  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile
import cv2
import numpy as np

import models
from models.experimental import attempt_load,End2End
from utils.activations import Hardswish, SiLU
from utils.general import colorstr, check_img_size, check_requirements, file_size, set_logging
from utils.torch_utils import select_device


def get_node_num(path,model=None):
    model = onnx.load(path) if model is None else model
    nodes = model.graph.node
    num = len(nodes)
    print(f'onnx node number: {num}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/yolov5l6_pose.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[832, 832], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')  # ONNX-only
    parser.add_argument('--half', action='store_true', help='FP16 precision')  # ONNX-only
    parser.add_argument('--simplify', action='store_true', help='simplify ONNX model')  # ONNX-only
    parser.add_argument('--end2end', action='store_true', help='simplify ONNX model')  # ONNX-only
    parser.add_argument('--topk-all', type=int, default=100, help='topk objects for every images')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='iou threshold for NMS')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='conf threshold for NMS')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    opt.dynamic = opt.dynamic and not opt.end2end 
    print(opt)
    set_logging()
    t = time.time()

    # Load PyTorch model
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    labels = model.names

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(device)  # image size(1,3,320,192) iDetection
 
    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
     
    model.eval()
    model.model[-1].export = True
    model.model[-1].onnx_dynamic = opt.dynamic
    model.model[-1].inplace = False
    model.model[-1].end2end = opt.end2end
    model = model.to(device)

    if opt.half:
        img = img.half()
        model = model.half()

    for _ in range(2):
        y = model(img)  # dry runs
    input_names = ['input']
    output_names = ['output']

    if opt.end2end:
        output_names = ['det_indices','det_boxes', 'det_pose', 'det_scores']
        model = End2End(model,opt.topk_all,opt.iou_thres,opt.conf_thres,device)
        shapes = [opt.batch_size,opt.topk_all, 3, opt.batch_size, opt.topk_all, 4,
                    opt.batch_size, opt.topk_all, 51, opt.batch_size, opt.topk_all,1]
        
    
    print(f"\n{colorstr('PyTorch:')} starting from {opt.weights} ({file_size(opt.weights):.1f} MB)")

    # ONNX export ------------------------------------------------------------------------------------------------------
    prefix = colorstr('ONNX:')
    try:
        warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)  # suppress TracerWarning

        import onnx
        print(f'{prefix} starting export with onnx {onnx.__version__}...')
        f = opt.weights.replace('.pt', '.onnx')  # filename
        if opt.end2end:
            f = f.replace('.onnx', '-NMS.onnx')
        if opt.dynamic:
            f = f.replace('.onnx', '-dynamic.onnx')
        torch.onnx.export(model, img, f, verbose=False, opset_version=11, 
                          input_names=input_names, output_names=output_names,
                          do_constant_folding=True if device.type == 'cpu' else False,
                          dynamic_axes={'input': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
                                        'output': {0: 'batch', 2: 'y', 3: 'x'}} if opt.dynamic else None)

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # print(onnx.helper.printable_graph(model_onnx.graph))  # print

        nodes = model_onnx.graph.node
        print(f'{prefix} onnx model node number: {len(nodes)}')

        if opt.end2end:
            for i in model_onnx.graph.output:
                for j in i.type.tensor_type.shape.dim:
                    j.dim_param = str(shapes.pop(0))
            onnx.save(model_onnx, f)
        # Simplify
        if opt.simplify:
            try:
                check_requirements(['onnx-simplifier'])
                import onnxsim

                print(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(model_onnx,
                                                     test_input_shapes={'input': list(img.shape)} if opt.dynamic else None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
                nodes = model_onnx.graph.node
                print(f'{prefix} onnx model node number after simplify: {len(nodes)}')
                # print(onnx.helper.printable_graph(model_onnx.graph))  # print
            except Exception as e:
                print(f'{prefix} simplifier failure: {e}')
        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
    except Exception as e:
        print(f'{prefix} export failure: {e}')
        traceback.print_exc()

    # Finish
    print(f'\nExport complete ({time.time() - t:.2f}s). Visualize with https://github.com/lutzroeder/netron.')
# python3 models/export_onnx.py --weights weights/yolov5s6_pose_ti_lite.pt --img 640 --end2end
