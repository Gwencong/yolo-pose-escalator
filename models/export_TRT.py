import os
import sys
import time
import argparse
import tensorrt as trt
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.absolute().__str__())  # to run '$ python *.py' files in subdirectories

from utils.general import colorstr, file_size, set_logging
from utils.torch_utils import select_device

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, default='weights/yolov5l6_pose_custom.onnx', help='onnx model path')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3')
    parser.add_argument('--fp16', action='store_true', help='export fp16 model')
    parser.add_argument('--int8', action='store_true', help='export int8 model')
    parser.add_argument('--workspace', type=int, default=4, help='maximum amount of persistent scratch memory available (in GB)')
    parser.add_argument('--verbose', action='store_true', help='print detail infomation of TRT export')
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes') # only when ONNX is dynamic
    parser.add_argument('--calib_path', type=str, default='data/custom_kpts/images', help='calibrate data path') 
    parser.add_argument('--calib_num', type=int, default=1024, help='calibration image number') 
    parser.add_argument('--calib_batch', type=int, default=128, help='clibration image batch size') 
    parser.add_argument('--calib_imgsz', type=int, default=832, help='clibration image size')
    parser.add_argument('--calib_method', type=str,choices=['MinMax','Entropy'], default='MinMax', help='calibration method')
    parser.add_argument('--calib_letterbox', action='store_true', help='whether letterbox when calibrate')
    parser.add_argument('--cache_dir', type=str, default='caches', help='cache file save directory')
    opt = parser.parse_args()
    print(opt)
    set_logging()
    device = select_device(opt.device,opt.batch_size)
    t = time.time()

    prefix = colorstr('TensorRT')
    print(f'\n{prefix} starting export with TensorRT {trt.__version__}...')

    verbose = opt.verbose
    workspace = opt.workspace
    half = opt.fp16 and not opt.int8
    int8 = opt.int8
    bs = opt.batch_size
    try:
        onnx_path = opt.onnx
        trt_path = opt.onnx.replace('.onnx','.trt')
        if half:
            trt_path = opt.onnx.replace('.onnx','-FP16.trt')
        elif int8:
            trt_path = opt.onnx.replace('.onnx','-INT8.trt')
        else:
            trt_path = opt.onnx.replace('.onnx','.trt')
        
        logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            logger.min_severity = trt.Logger.Severity.VERBOSE

        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        config.max_workspace_size = workspace * 1 << 30
        # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation notice

        flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(onnx_path):
            raise RuntimeError(f'failed to load ONNX file: {onnx_path}')
        
        if opt.dynamic:
            profile = builder.create_optimization_profile()     
            profile.set_shape("input", (bs, 3, 320, 320), (bs, 3, 640, 640), (bs, 3, 1280, 1280))
            config.add_optimization_profile(profile)

        # unmark unnecessary output layers
        # count = 0
        # while (network.num_outputs>1):
        #     out = network.get_output(count)
        #     if out.name!='output':
        #         network.unmark_output(out)
        #         print(f'{prefix}\toutput "{out.name}" with shape {out.shape} and dtype {out.dtype} has been removed')
        #     else:
        #         count+=1

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        onnx_input_dtype = inputs[0].dtype
        print(f'{prefix} Network Description:')
        for inp in inputs:
            print(f'{prefix}\tinput "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}')
        for out in outputs:
            print(f'{prefix}\toutput "{out.name}" with shape {out.shape} and dtype {out.dtype}')

        if not builder.platform_has_fast_fp16:
            print(colorstr('bold','red','Warning: FP16 is not supported on this platform!'))
        if not builder.platform_has_fast_int8:
            print(colorstr('bold','red','Warning: INT8 is not supported on this platform!'))

        precision = 'FP32'
        if builder.platform_has_fast_fp16 and half:
            precision = 'FP16'
            config.set_flag(trt.BuilderFlag.FP16)
        elif builder.platform_has_fast_int8 and int8:
            from utils.calibrator import get_int8_calibrator
            precision = 'INT8'
            config.flags |= 1 << int(trt.BuilderFlag.INT8)
            config.flags |= 1 << int(trt.BuilderFlag.FP16)
            Path(opt.cache_dir).mkdir(parents=True,exist_ok=True)
            calib_cache = Path(os.path.join(opt.cache_dir,os.path.basename(onnx_path))).with_suffix('.cache')
            config.int8_calibrator = get_int8_calibrator(calib_path   = opt.calib_path, 
                                                         calib_batch  = opt.calib_batch,
                                                         calib_num    = opt.calib_num,
                                                         img_size     = opt.calib_imgsz,
                                                         cache_file   = str(calib_cache),
                                                         calib_method = opt.calib_method,
                                                         letterbox    = opt.calib_letterbox,
                                                         half = onnx_input_dtype == trt.DataType.HALF)
        print(f'{prefix} building {precision} engine in {trt_path}')
        with builder.build_engine(network, config) as engine, open(trt_path, 'wb') as f:
            f.write(engine.serialize())
        print(f'{prefix} export success, saved as {trt_path} ({file_size(trt_path):.1f} MB)')
        print(f'\nExport complete ({time.time() - t:.2f}s). ')
    except Exception as e:
        print(f'\n{prefix} export failure: {e}')

# python3 models/export_TRT.py --onnx weights/yolov5s6_pose_ti_lite.onnx 