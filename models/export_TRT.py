import sys
import time
import argparse
import tensorrt as trt
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.absolute().__str__())  # to run '$ python *.py' files in subdirectories

from utils.general import colorstr, file_size, set_logging



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, default='weights/yolov5s6_pose_640_ti_lite.onnx', help='onnx model path')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--fp16', default=True, help='export fp16 model')
    parser.add_argument('--workspace', type=int, default=4, help='maximum amount of persistent scratch memory available (in GB)')
    parser.add_argument('--verbose', default=True, help='print detail infomation of TRT export')
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes') # only when ONNX is dynamic
    opt = parser.parse_args()
    print(opt)
    set_logging()
    t = time.time()

    prefix = colorstr('TensorRT')
    print(f'\n{prefix} starting export with TensorRT {trt.__version__}...')

    verbose = opt.verbose
    workspace = opt.workspace
    half = opt.fp16
    bs = opt.batch_size
    try:
        onnx_path = opt.onnx
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
            profile.set_shape("input", (bs, 3, 256, 256), (bs, 3, 640, 640), (bs, 3, 1280, 1280))
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
        print(f'{prefix} Network Description:')
        for inp in inputs:
            print(f'{prefix}\tinput "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}')
        for out in outputs:
            print(f'{prefix}\toutput "{out.name}" with shape {out.shape} and dtype {out.dtype}')

        print(f'{prefix} building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine in {trt_path}')
        if builder.platform_has_fast_fp16 and half:
            config.set_flag(trt.BuilderFlag.FP16)
        with builder.build_engine(network, config) as engine, open(trt_path, 'wb') as f:
            f.write(engine.serialize())
        print(f'{prefix} export success, saved as {trt_path} ({file_size(trt_path):.1f} MB)')
        print(f'\nExport complete ({time.time() - t:.2f}s). ')
    except Exception as e:
        print(f'\n{prefix} export failure: {e}')

# python3 models/export_TRT.py --onnx weights/yolov5s6_pose_ti_lite.onnx 