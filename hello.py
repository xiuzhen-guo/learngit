import numpy as np
import cv2
import random
from skimage.io import imread
from skimage import color

import torch
import torch.utils.data
from torchvision import datasets,models,transforms

class Dataset(torch.utils.data.Dataset):

    def  __init__(self,img_paths,mask_paths,imgwbq_paths,aug=False):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.imgwbq_paths = imgwbq_paths
        self.aug = aug

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self,idx):
        if idx in range(len(self.imgwbq_paths)-1):
            print('self.imgwbq_paths',self.imgwbq_paths[idx])
            print('self.img_paths', self.img_paths[idx])
            global imgwbq_path
            imgwbq_path = self.imgwbq_paths[idx]
        else:
            print('self.img_paths', self.img_paths[idx])
        #print('self.img_paths',self.img_paths[idx])
        #print('self.imgwbq_paths',self.imgwbq_paths[idx])
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        #imgwbq_path = self.imgwbq_paths[idx]
        # 读numpy数据(npy)的代码
        npimage = np.load(img_path)
        npmask =  np.load(mask_path)
        npimagewbq = np.load(imgwbq_path)

        npmask = npmask.astype(str)
        npimage = npimage.astype(str)
        npimagewbq = npimagewbq.astype(str)



        return {
                     'image':torch.as_tensor(npimage.copy()).contiguous(),
                   'label':torch.as_tensor(npmask.copy()).contiguous(),
                    'imagewbq': torch.as_tensor(npimagewbq.copy()).contiguous()
                }



    '''def runnum(self):
        a=len(self.img_paths)
        b=len(self.imgwbq_paths)
        c=len(self.mask_paths)
        print(a)
        print(b)
        print(c)'''


'''from  glob import glob
img_paths = glob(r'/home/xiuzhen-guo/datasets/BraTS20193D/trainImage/*')
mask_paths = glob(r'/home/xiuzhen-guo/datasets/BraTS20193D/trainMask/*')
imgwbq_paths = glob(r'/home/xiuzhen-guo/datasets/BraTS20193D/trainImagewbq/*')
train_dataset = Dataset(img_paths,mask_paths,imgwbq_paths)
#train_dataset.rundata()
train_dataset.runnum()'''

from torch.utils.data.sampler import Sampler
import itertools
class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)






