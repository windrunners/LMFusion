import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os,sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from metrics import *
from option import opt
BS=opt.bs
print(BS)
crop_size='whole_img'
if opt.crop:
    crop_size=opt.crop_size

def tensorShow(tensors,titles=None):
        '''
        t:BCWH
        '''
        fig=plt.figure()
        for tensor,tit,i in zip(tensors,titles,range(len(tensors))):
            img = make_grid(tensor)
            npimg = img.numpy()
            ax = fig.add_subplot(211+i)
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(tit)
        plt.show()

class RESIDE_Dataset(data.Dataset):
    def __init__(self, path, train, size=crop_size, format='.png'):
        super(RESIDE_Dataset, self).__init__()
        self.size = size
        print('crop size', size)
        self.train = train
        self.format = format
        self.haze_imgs_dir = os.listdir(os.path.join(path, 'hazy'))  # connect to the next level directory
        self.haze_imgs = [os.path.join(path, 'hazy', img) for img in self.haze_imgs_dir]
        self.clear_dir = os.path.join(path, 'clear')
    def __getitem__(self, index):
        haze = Image.open(self.haze_imgs[index])
        if isinstance(self.size, int):   # size is int and returns true
            while haze.size[0] < self.size or haze.size[1] < self.size:
                index = random.randint(0, 20000)
                haze = Image.open(self.haze_imgs[index])
        img = self.haze_imgs[index]
        id = img.split('\\')[-1].split('_')[0]
        clear_name = id[0:4]+self.format
        clear = Image.open(os.path.join(self.clear_dir, clear_name))
        clear = tfs.CenterCrop(haze.size[::-1])(clear)
        if not isinstance(self.size, str):
            i, j, h, w = tfs.RandomCrop.get_params(haze, output_size=(self.size, self.size))
            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)
        haze = haze.convert("RGB")        # if png, RGBA to RGB
        clear = clear.convert("RGB")      # if png, RGBA to RGB
        haze_numpy = np.array(haze) / 255.0
        clear_numpy = np.array(clear) / 255.0
        haze1, haze1_cb, haze1_cr = self.rgb2ycbcr(haze_numpy)
        clear1, clear1_cb, clear1_cr = self.rgb2ycbcr(clear_numpy)
        haze2 = haze1.astype(np.float32)
        clear2 = clear1.astype(np.float32)
        haze3, clear3 = self.augData(haze2, clear2)
        return haze3, clear3
    def augData(self, data, target):
        # if self.train:
            # rand_hor = random.randint(0, 1)
            # rand_rot = random.randint(0, 3)
            # data = tfs.RandomHorizontalFlip(rand_hor)(data)
            # target = tfs.RandomHorizontalFlip(rand_hor)(target)
            # if rand_rot:
                # data = FF.rotate(data, 90*rand_rot)
                # target = FF.rotate(target, 90*rand_rot)
        data = tfs.ToTensor()(data)
        # data = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(data)
        # data = tfs.Normalize(mean=[0.6], std=[0.15])(data)
        target = tfs.ToTensor()(target)
        return data, target
    def __len__(self):
        return len(self.haze_imgs)

    def rgb2ycbcr(self, img_rgb):
        R = img_rgb[:, :, 0]
        G = img_rgb[:, :, 1]
        B = img_rgb[:, :, 2]
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128 / 255.0
        Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128 / 255.0
        return Y, Cb, Cr

    def ycbcr2rgb(self, Y, Cb, Cr):
        R = Y + 1.402 * (Cr - 128 / 255.0)
        G = Y - 0.34414 * (Cb - 128 / 255.0) - 0.71414 * (Cr - 128 / 255.0)
        B = Y + 1.772 * (Cb - 128 / 255.0)
        R = np.expand_dims(R, axis=-1)
        G = np.expand_dims(G, axis=-1)
        B = np.expand_dims(B, axis=-1)
        return np.concatenate([R, G, B], axis=-1)

import os
pwd=os.getcwd()
print(pwd)
path='home/data'  # path to your data folder

ITS_train_loader=DataLoader(dataset=RESIDE_Dataset(path+'/RESIDE/ITS',train=True,size=crop_size),batch_size=BS,shuffle=True)
ITS_test_loader=DataLoader(dataset=RESIDE_Dataset(path+'/RESIDE/SOTS/indoor',train=False,size=crop_size),batch_size=1,shuffle=False)

OTS_train_loader=DataLoader(dataset=RESIDE_Dataset(path+'/RESIDE/OTS',train=True,format='.jpg'),batch_size=BS,shuffle=True)
OTS_test_loader=DataLoader(dataset=RESIDE_Dataset(path+'/RESIDE/SOTS/indoor',train=False,size=crop_size,format='.png'),batch_size=1,shuffle=False)

if __name__ == "__main__":
    pass
