import os
import cv2
import glob
import torch
import random
import numpy as np
import tensorrt as trt


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


class MinMaxCalibrator_torch(trt.IInt8MinMaxCalibrator):

    def __init__(self, calibrationDataPath, calibrationCount, inputShape, cacheFile,letterbox=False, half=False):
        super().__init__()
        assert os.path.exists(calibrationDataPath),f'calibration path `{calibrationDataPath}` not exist'
        self.imageList = self.get_imagesList(calibrationDataPath,inputShape[0],calibrationCount)
        self.letterbox = letterbox
        self.calibrationCount = calibrationCount
        self.shape = inputShape  # (N,C,H,W)
        self.cacheFile = cacheFile
        self.half = half

        self.device = torch.device('cuda:0')
        self.dtype = trt.nptype(trt.float16) if half else trt.nptype(trt.float32)
        self.device_mem = torch.from_numpy(np.empty(self.shape, dtype=np.dtype(self.dtype))).to(self.device)
        self.dIn = self.device_mem.data_ptr()

        self.oneBatch = self.batchGenerator()

        print(int(self.dIn))
        print(self.get_algorithm())
        
    def __del__(self):
        del self.device_mem
    
    def get_imagesList(self,calib_path,batch_size,calib_num):
        all_images = glob.glob(os.path.join(calib_path, "*.jpg"))
        sample_num = calib_num - len(all_images)
        if len(all_images) == 0:
            print(f'Calib: Find {len(all_images)} images, need calib cache file')
            imageList = []
        elif sample_num>0:
            print(f'Calib: Find {len(all_images)} images, random resampling {sample_num} images, total use {calib_num}')
            sampleList = np.random.choice(all_images, sample_num, replace=False).tolist()
            imageList = all_images + sampleList
        else:
            print(f'Calib: Find {len(all_images)} images, total use {calib_num}')
            imageList = all_images[:calib_num]
        random.shuffle(imageList)
        return imageList
        

    def batchGenerator(self):
        for i in range(0,self.calibrationCount,self.shape[0]):
            print("> calibration %d/%d" % (i+self.shape[0],self.calibrationCount))
            # subImageList = np.random.choice(self.imageList, self.shape[0], replace=False)
            subImageList = self.imageList[i:i+self.shape[0]]
            yield np.ascontiguousarray(self.loadImageList(subImageList))

    def loadImageList(self, imageList):
        res = np.empty(self.shape, dtype=np.dtype(self.dtype))
        for i in range(self.shape[0]):
            img = cv2.imread(imageList[i])
            if self.letterbox:
                img = letterbox(img,new_shape=self.shape[2:],auto=True,stride=32)[0]
            img = cv2.resize(img,dsize=(self.shape[2],self.shape[3]),interpolation=cv2.INTER_LINEAR)
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = img/255.0
            res[i] = img.astype(np.dtype(self.dtype))

        return res

    def get_batch_size(self):  # do NOT change name
        return self.shape[0]

    def get_batch(self, nameList=None, inputNodeName=None):  # do NOT change name
        try:
            data = next(self.oneBatch)
            self.device_mem = torch.from_numpy(data.astype(np.dtype(self.dtype))).to(self.device)
            return [self.device_mem.data_ptr()]
        except StopIteration:
            return None

    def read_calibration_cache(self):  # do NOT change name
        if os.path.exists(self.cacheFile):
            print("Succeed finding cahce file: %s" % (self.cacheFile))
            with open(self.cacheFile, "rb") as f:
                cache = f.read()
                return cache
        else:
            print("Failed finding int8 cache!")
            return

    def write_calibration_cache(self, cache):  # do NOT change name
        with open(self.cacheFile, "wb") as f:
            f.write(cache)
        print("Succeed saving int8 cache!")

class EntropyCalibrator_torch(trt.IInt8EntropyCalibrator2):

    def __init__(self, calibrationDataPath, calibrationCount, inputShape, cacheFile,letterbox=False,half=False):
        super().__init__()
        assert os.path.exists(calibrationDataPath),f'calibration path `{calibrationDataPath}` not exist'
        self.imageList = self.get_imagesList(calibrationDataPath,inputShape[0],calibrationCount)
        self.letterbox = letterbox
        self.calibrationCount = calibrationCount
        self.shape = inputShape  # (N,C,H,W)
        self.cacheFile = cacheFile
        self.half = half

        self.device = torch.device('cuda:0')
        self.dtype = trt.nptype(trt.float16) if half else trt.nptype(trt.float32)
        self.device_mem = torch.from_numpy(np.empty(self.shape, dtype=np.dtype(self.dtype))).to(self.device)
        self.dIn = self.device_mem.data_ptr()

        self.oneBatch = self.batchGenerator()

        print(int(self.dIn))
        print(self.get_algorithm())
        
    def __del__(self):
        del self.device_mem
    
    def get_imagesList(self,calib_path,batch_size,calib_num):
        all_images = glob.glob(os.path.join(calib_path, "*.jpg"))
        sample_num = calib_num - len(all_images)
        if len(all_images) == 0:
            print(f'Calib: Find {len(all_images)} images, need calib cache file')
            imageList = []
        elif sample_num>0:
            print(f'Calib: Find {len(all_images)} images, random resampling {sample_num} images, total use {calib_num}')
            sampleList = np.random.choice(all_images, sample_num, replace=False).tolist()
            imageList = all_images + sampleList
        else:
            print(f'Calib: Find {len(all_images)} images, total use {calib_num}')
            imageList = all_images[:calib_num]
        random.shuffle(imageList)
        return imageList
        

    def batchGenerator(self):
        for i in range(0,self.calibrationCount,self.shape[0]):
            print("> calibration %d/%d" % (i+self.shape[0],self.calibrationCount))
            # subImageList = np.random.choice(self.imageList, self.shape[0], replace=False)
            subImageList = self.imageList[i:i+self.shape[0]]
            yield np.ascontiguousarray(self.loadImageList(subImageList))

    def loadImageList(self, imageList):
        res = np.empty(self.shape, dtype=np.dtype(self.dtype))
        for i in range(self.shape[0]):
            img = cv2.imread(imageList[i])
            if self.letterbox:
                img = letterbox(img,new_shape=self.shape[2:],auto=True,stride=64)[0]
            img = cv2.resize(img,dsize=(self.shape[2],self.shape[3]),interpolation=cv2.INTER_LINEAR)
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = img/255.0
            res[i] = img.astype(self.dtype)

        return res

    def get_batch_size(self):  # do NOT change name
        return self.shape[0]

    def get_batch(self, nameList=None, inputNodeName=None):  # do NOT change name
        try:
            data = next(self.oneBatch)
            self.device_mem = torch.from_numpy(data.astype(np.dtype(self.dtype))).to(self.device)
            return [self.device_mem.data_ptr()]
        except StopIteration:
            return None

    def read_calibration_cache(self):  # do NOT change name
        if os.path.exists(self.cacheFile):
            print("Succeed finding cahce file: %s" % (self.cacheFile))
            with open(self.cacheFile, "rb") as f:
                cache = f.read()
                return cache
        else:
            print("Failed finding int8 cache!")
            return

    def write_calibration_cache(self, cache):  # do NOT change name
        with open(self.cacheFile, "wb") as f:
            f.write(cache)
        print("Succeed saving int8 cache!")


def get_int8_calibrator(calib_path, calib_batch, calib_num, img_size, cache_file, calib_method='MinMax', letterbox=False, half=False):
    methods = ['MinMax','Entropy']
    assert calib_method in methods,f'method shoud be one of {methods}'
    inputShape = (calib_batch,3,img_size,img_size)
    if calib_method == methods[0]:
        calibrator = MinMaxCalibrator_torch(calib_path, calib_num, inputShape, cache_file, letterbox, half)
    else:
        calibrator = EntropyCalibrator_torch(calib_path, calib_num, inputShape, cache_file, letterbox, half)
    return calibrator



if __name__ == "__main__":
    calib_path  = "./data/custom_kpts/images"
    calib_batch = 10
    calib_num   = 50
    img_size    = 640
    cache_file  = "./caches/int8.cache"
    calib_method= 'MinMax'
    letter      = False
    m = get_int8_calibrator(calib_path, calib_batch, calib_num, img_size, cache_file, calib_method,letter,half=True)
    for i in range(int(calib_num/calib_batch)):
        m.get_batch("FakeNameList")
