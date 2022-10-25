import os 
import sys
import cv2
import numpy as np
from pathlib import Path
from collections import OrderedDict,namedtuple

import tensorrt as trt
import torch
import torch.nn as nn
import onnx 
import onnxruntime

sys.path.append(Path(__file__).parent.parent.absolute().resolve().__str__())

from utils.general import colorstr
from models.common import Conv
from models.experimental import attempt_load
from utils.activations import Hardswish,SiLU

class TRT_Infer():
    def __init__(self, engine_path, device, input_shape=None,conf=0.4,iou=0.5,engine=None,stride=64) -> None:
        self.iou = iou
        self.conf = conf
        self.device = device
        self.stride = stride
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(engine_path,self.runtime) if engine is None else engine
        self.context = self.engine.create_execution_context()
        self.input_shape, self.dynamic = self.get_shape(self.engine,input_shape)
        self.bindings, self.input_dtype = self.allocate_buffers(self.context,self.input_shape)
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())

        if 'float16' in str(self.input_dtype):
            self.half = True
        else:
            self.half = False
        

    def __call__(self, img_src, *args, **kwds):
        b,c,h,w = self.input_shape
        # pytorch 传输数据到tensorrt时，输入数据类型必须与模型输入数据类型一致，否者无法产生正确推理结果
        if isinstance(img_src,np.ndarray):
            img_src = img_src.astype(self.input_dtype)
            img_src = torch.from_numpy(img_src)
        else:
            img_src = img_src.half() if self.half else img_src.float()
            
        img = img_src.to(self.device)
        out = self.inference(img)
        if len(out)>1:
            nmsed_indices,nmsed_boxes,nmsed_poses,nmsed_scores = out
            nmsed_indices = nmsed_indices.reshape(b,-1,3)
            nmsed_boxes = nmsed_boxes.reshape(b,-1,4)
            nmsed_poses = nmsed_poses.reshape(b,-1,51)
            nmsed_scores = nmsed_scores.reshape(b,-1,1)
            nmsed_confes = torch.ones_like(nmsed_scores).to(nmsed_scores.device)
            keep = torch.unique(nmsed_indices[...,2]).numel()
            if torch.any(torch.isnan(nmsed_indices[...,2])) or torch.all(nmsed_indices[...,2] < 0):
                keep = 0
            out = torch.cat([nmsed_boxes,nmsed_scores,nmsed_confes,nmsed_poses],axis=-1)
            out = out[:,:keep,:]
        else:
            out = out[0]
        out = out.reshape(b,-1,57)
        return out


    def load_engine(self,engine_path,runtime):
        trt.init_libnvinfer_plugins(None,'')    # load all avalilable official pluging
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        return engine

    def warmup(self,times=5):
        # print(f'warmup {times} times...')
        b,c,h,w = self.input_shape
        dummyinput = np.random.randn((h,w,c))
        # dummyinput = self.preprocess(dummyinput)
        for i in range(times):
            self.inference(dummyinput)

    def get_shape(self,engine,input_shape=None):
        dynamic = False
        shape = engine.get_binding_shape(0)
        if -1 in shape:
            dynamic = True
            shapes = engine.get_profile_shape(profile_index=0, binding=0)
            min_shape,opt_shape,max_shape = shapes
            shape = input_shape
            print(f'Engine shape range:\n  MIN: {min_shape}\n  OPT: {opt_shape}\n  MAX: {max_shape}')
            print(f'Set engine shape to: {input_shape}')
        else:
            assert shape == input_shape, f'engine shape `{shape}` is not compatible with given shape `{input_shape}`'
            print(f'Engine input shape:  {shape}')
        return shape,dynamic

    def allocate_buffers(self,context,shape):
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr', 'is_input'))
        bindings = OrderedDict()
        if self.dynamic:
            context.set_binding_shape(0,shape)   # Dynamic Shape 模式需要绑定真实数据形状
        engine = context.engine
        print('\n{:<20s}{:^30s}{:^20s}{:>20s}'.format('Binding name', 'dtype', 'shape', 'is_input'))
        for binding in engine:
            ind   = engine.get_binding_index(binding)
            name  = engine.get_binding_name(ind)
            dtype = trt.nptype(engine.get_binding_dtype(ind))
            shape = tuple(context.get_binding_shape(ind))
            data  = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(self.device)
            is_input = engine.binding_is_input(ind)
            if is_input:
                input_dtype = dtype
            bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()), is_input)
            print('{:<20s}{:^30s}{:^20s}{:>20s}'.format(name, str(dtype), str(shape), str(is_input)))
        print('')
        return bindings, input_dtype


    def inference(self,img):
        self.binding_addrs[self.engine.get_binding_name(0)] = int(img.data_ptr())
        self.context.execute_v2(bindings=list(self.binding_addrs.values()))
        output = []
        for name in self.bindings:
            if not self.bindings[name].is_input:
                output.append(self.bindings[name].data)
        return output
    

class Pt_Infer():
    def __init__(self,weight,device,half,fuse=True) -> None:
        self.weight = weight
        self.device = device
        self.half = half
        self.fuse = fuse

        model = attempt_load(weight, map_location=device)  # load FP32 model
        for k, m in model.named_modules():
            m._non_persistent_buffers_set = set()   # pytorch 1.6.0 compatibility
            if isinstance(m, Conv):                 # assign export-friendly activations
                if isinstance(m.act, nn.Hardswish):
                    m.act = Hardswish()
                elif isinstance(m.act, nn.SiLU):
                    m.act = SiLU()
        if half:
            model.half()                    # to FP16
        model.to(device)
        stride = int(model.stride.max())    # model stride

        self.model = model
        self.stride = stride

    def __call__(self, img, *args, **kwds):
        if isinstance(img,np.ndarray):
            img = torch.from_numpy(img)
        img = img.to(self.device)
        img = img.half() if self.half else img.float()
        output = self.model(img,*args, **kwds)[0]
        return output


class ONNX_Infer():
    def __init__(self,onnx_path,device,half=False,stride=64) -> None:
        self.path = onnx_path
        self.device = device
        self.stride = stride
        self.provider = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(onnx_path,providers=self.provider)
        self.input_name = self.session.get_inputs()[0].name
        self.half = self.session.get_inputs()[0].type == 'tensor(float16)'
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        self.check()

    def check(self):
        if self.device.type != 'cpu':
            try:
                img = np.random.randn(self.input_shape)
                inputs = {self.input_name: img} 
                self.session.run([self.output_name], inputs)
            except:
                print(f'Infernce with CUDA error for some reasons, set to use CPU')
                self.provider = ['CPUExecutionProvider']
                self.session = onnxruntime.InferenceSession(self.path,providers=self.provider)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name

    def warmup(self,times):
        for i in range(times):
            img = np.random.randn(self.input_shape)
            inputs = {self.input_name: img} 
            self.session.run([self.output_name], inputs)

    def __call__(self, img,*args, **kwds):
        if isinstance(img,torch.Tensor):
            img = img.cpu().numpy()
        img = img.astype(np.float16 if self.half else np.float32) 
        inputs = {self.input_name: img}
        outputs = self.session.run([self.output_name], inputs)
        output = torch.from_numpy(outputs[0]).float()
        return output


class TRT_Infer_pycuda():
    
    class HostDeviceMem(object):
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

        def __str__(self):
            return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

        def __repr__(self):
            return self.__str__()

    def __init__(self, engine_path, device, input_shape=None,conf=0.4,iou=0.5,engine=None,stride=64) -> None:
        import pycuda.autoinit
        import pycuda.driver as cuda
        self.iou = iou
        self.conf = conf
        self.device = device
        self.stride = stride
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(engine_path,self.runtime) if engine is None else engine
        self.context = self.engine.create_execution_context()
        self.input_shape, self.dynamic = self.get_shape(self.engine,input_shape)
        self.inputs, self.outputs, self.bindings, self.stream, self.input_dtype = \
            self.allocate_buffers(self.context,self.input_shape)
        
        if 'float16' in str(self.input_dtype):
            self.half = True
        else:
            self.half = False
        

    def __call__(self, img_src, *args, **kwds):
        b,c,h,w = self.input_shape
        # pytorch 传输数据到tensorrt时，输入数据类型必须与模型输入数据类型一致，否者无法产生正确推理结果
        if not isinstance(img_src,np.ndarray):
            img_src = img_src.cpu().numpy()
        img = img_src.astype(self.input_dtype)

        out = self.inference(img)
        if len(out)>1:
            nmsed_indices,nmsed_boxes,nmsed_poses,nmsed_scores = out
            nmsed_indices = nmsed_indices.reshape(b,-1,3)
            nmsed_boxes = nmsed_boxes.reshape(b,-1,4)
            # nmsed_boxes[0] = self.xywh2xyxy(nmsed_boxes[0])
            nmsed_poses = nmsed_poses.reshape(b,-1,51)
            nmsed_scores = nmsed_scores.reshape(b,-1,1)
            nmsed_confes = np.ones_like(nmsed_scores)
            keep = np.unique(nmsed_indices[...,2]).size
            out = np.concatenate([nmsed_boxes,nmsed_scores,nmsed_confes,nmsed_poses],axis=-1)
            out = out[:,:keep,:]
        else:
            out = out[0]
        out = out.reshape(b,-1,57)
        out = torch.from_numpy(out).float()
        return out

    def load_engine(self,engine_path,runtime):
        trt.init_libnvinfer_plugins(None,'')    # load all avalilable official pluging
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        return engine

    def warmup(self,times=5):
        # print(f'warmup {times} times...')
        b,c,h,w = self.input_shape
        dummyinput = np.random.randn((h,w,c))
        # dummyinput = self.preprocess(dummyinput)
        for i in range(times):
            self.inference(dummyinput)

    def get_shape(self,engine,input_shape=None):
        dynamic = False
        shape = engine.get_binding_shape(0)
        if -1 in shape:
            dynamic = True
            shapes = engine.get_profile_shape(profile_index=0, binding=0)
            min_shape,opt_shape,max_shape = shapes
            shape = input_shape
            print(f'Engine shape range:\n  MIN: {min_shape}\n  OPT: {opt_shape}\n  MAX: {max_shape}')
            print(f'Set engine shape to: {input_shape}')
        else:
            assert shape == input_shape, f'engine shape `{shape}` is not compatible with given shape `{input_shape}`'
            print(f'Engine input shape:  {shape}')
        return shape,dynamic

    def allocate_buffers(self,context,shape):
        inputs = []
        outputs = []
        bindings = []
        inp_dtype = trt.nptype(trt.float32)
        stream = cuda.Stream()
        context.set_binding_shape(0,shape)   # Dynamic Shape 模式需要绑定真实数据形状
        engine = context.engine
        print('\n{:<20s}{:^30s}{:^20s}{:>20s}'.format('Binding name', 'dtype', 'shape', 'is_input'))
        for binding in engine:
            ind = engine.get_binding_index(binding)
            name = engine.get_binding_name(ind)
            shape = tuple(context.get_binding_shape(ind))
            size = trt.volume(context.get_binding_shape(ind)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(ind))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            is_input = engine.binding_is_input(ind)
            if is_input:
                inputs.append(self.HostDeviceMem(host_mem, device_mem))
                inp_dtype = dtype
            else:
                outputs.append(self.HostDeviceMem(host_mem, device_mem))
            print('{:<20s}{:^30s}{:^20s}{:>20s}'.format(name, str(dtype), str(shape), str(is_input)))
        return inputs, outputs, bindings, stream, inp_dtype


    def inference(self,img):
        np.copyto(self.inputs[0].host, (img.astype(self.input_dtype).ravel()))
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        self.stream.synchronize()
        output = [out.host for out in self.outputs]
        return output

    def xywh2xyxy(self,x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y



def load_model(weight,device,half=False,input_shape=None,fuse=True):
    if isinstance(weight,list):
        weight = weight[0]
    if weight.endswith('.pt'):
        prefix = colorstr('Pytorch')
        print(f'{prefix}: load model from {os.path.abspath(weight)}...')
        assert os.path.exists(weight), f'{prefix}: model file not exist'
        model = Pt_Infer(weight,device,half,fuse)
    elif weight.endswith('.onnx'):
        prefix = colorstr('ONNX')
        print(f'{prefix}: load model from {os.path.abspath(weight)}...')
        node_nums = len(onnx.load(weight).graph.node)
        print(f'{prefix}: model nodes: {node_nums}')
        model = ONNX_Infer(weight,device,half)
    else:
        prefix = colorstr('TensorRT')
        print(f'{prefix}: load model from {os.path.abspath(weight)}...')
        model = TRT_Infer(weight,device,input_shape)
    print(f'{prefix}: load model successfully')
    return model

if __name__ == "__main__":
    from utils.general import non_max_suppression, scale_coords
    from utils.plots import plot_one_box
    from utils.datasets import letterbox
    device = torch.device("cuda:0")
    im0 = cv2.imread('data/images/bus.jpg')
    img = letterbox(im0,(832,832),auto=False,stride=64)[0]
    img = np.ascontiguousarray(img[:,:,::-1].transpose(2,0,1))
    img = np.expand_dims(img, 0)
    img = img.astype(np.float32)
    img = torch.from_numpy(img)
    img /= 255.0
    model = load_model("weights/yolov5l6_pose-NMS-FP16.trt",torch.device("cuda:0"),False,(1,3,832,832))
    out = model(img)
    pred = non_max_suppression(out, 0.25, 0.45,kpt_label=True)
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
            scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=True, step=3)

            # Write results
            for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:,:6])):
                label = None
                kpts = det[det_index, 6:]
                plot_one_box(xyxy, im0, label=label, kpt_label=True, kpts=kpts, steps=3, orig_shape=im0.shape[:2])
            save_path = 'test.jpg'
            cv2.imwrite(save_path,im0)
            print(f'image has been saved to {os.path.abspath(save_path)}')
        else:
            print('no results')

