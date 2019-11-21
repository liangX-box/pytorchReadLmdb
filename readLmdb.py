import sys
caffe_root = "/usr/caffe/"; sys.path.insert(0, caffe_root+"python")
import cv2, os, math, caffe, lmdb, io
import numpy as np
from caffe.proto import caffe_pb2
from PIL import Image
from io import BytesIO

class readLmdb:
    def __init__(self, key_path_, lmdb_path_, batch_size_, train=False):
        self.key_path = key_path_
        self.lmdb_path = lmdb_path_
        self.batch_size = batch_size_
        self.key_lt = open(self.key_path).read().splitlines()
        self.env = lmdb.open(self.lmdb_path)
        self.txn = self.env.begin()
        self.dataset_size = len(self.key_lt)
        print ("the size of samples is: %d" %(self.dataset_size))
        self.indices_total = np.arange(self.dataset_size)
        np.random.seed(321)
        np.random.shuffle(self.indices_total)
        self.data_idx = 0
        iter_batch = int(self.dataset_size / self.batch_size)
        self.splitNum = iter_batch * self.batch_size
        self.splitSize = self.dataset_size - self.splitNum
        self.splitBatch = 0
    def GetBatch(self):
        image_batch = np.zeros((self.batch_size, 1, 224, 224), dtype=np.float32)
        label_batch = np.zeros((self.batch_size), dtype=getattr(np, 'long'))
        for i in range(self.batch_size):
            ind = self.indices_total[self.data_idx]
            self.data_idx += 1
            if self.data_idx == self.dataset_size:
                self.data_idx = 0
                np.random.shuffle(self.indices_total)
            temp = (self.key_lt[ind]).encode()
            value = self.txn.get(temp)
            datum = caffe_pb2.Datum()
            datum.ParseFromString(value)
            label = float(datum.label)
            encoded = datum.encoded
            if encoded:
                stream = BytesIO(datum.data)
                img = np.uint8(Image.open(stream))
                img = img[...,::-1]
            else:
                data = caffe.io.datum_to_array(datum)
                img = np.transpose(data, (1, 2, 0))
            img_tmp = img.copy()
            img_tmp = np.float64(img_tmp)
            img_tmp -= 127.5
            img_tmp *= 0.0078125
            img_tmp = img_tmp[np.newaxis, :]
            image_batch[i, :] = img_tmp
            label_batch[i] = label
        return image_batch, label_batch