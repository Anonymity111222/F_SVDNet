from argparse import Namespace
from collections import Counter
import csv
import gc
from itertools import product
from re import S
from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
from numpy import array
import os
import pandas as pd
from tqdm import tqdm_notebook as tqdm

class FFT1D_block(nn.Module):
    def __init__(self, in_channels, out_channels, modes1,modes2, num=None):
        super(FFT1D_block, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.out_channels = out_channels
        self.num = num
        self.scale = (1 / (in_channels*out_channels))
        self.weights_FFT = nn.Parameter(self.scale * torch.rand(modes1,modes2, dtype=torch.cfloat))
       
    def fft_1D(self, x):
        if self.num:
            x_ft = torch.fft.fft(x,dim=1)
            x_ft = torch.einsum("bix,io->box",x_ft , self.weights_FFT)
            x = torch.fft.ifft(x_ft,dim=1, n=self.num)
            x = x.real
        else:
            x_ft = torch.fft.fft(x,dim=1)
            x_ft = torch.einsum("bix,io->box",x_ft , self.weights_FFT)
            x = torch.fft.ifft(x_ft,dim=1, n=self.out_channels)
            x = x.real
        return x
    
    def forward(self, x):
        x = self.fft_1D(x)
        return x


class SVD_Block_3(nn.Module):
    def __init__(self, in_channels, out_channels, modes1,modes2, F_channels):
        super(SVD_Block_3, self).__init__()
       
        self.W_FFT1 = FFT1D_block(in_channels, out_channels, modes1,modes2)
        self.weight_SVD = nn.Parameter(torch.rand(in_channels, F_channels))
        self.drop = nn.Dropout(0.5)
    def forward(self, x):
        u,s,v = torch.linalg.svd(x,full_matrices=False)
        wu,ws,wv = torch.linalg.svd(self.weight_SVD,full_matrices=False)
        s = ws*s
        s = torch.diag_embed(s)
      
        u = u*wu
        v = v*wv
        
        out_us = torch.matmul(u,s)
        out_svd = torch.matmul(out_us,v)
        out_svd = self.drop(torch.sin(out_svd))

        out_1 = self.W_FFT1(x)
        out_1 = out_1 + out_svd
        return out_1

class F_SVD_Net_3(nn.Module):
    def __init__(self, modes, channels, pred_num, F_channels):
        super(F_SVD_Net_3, self).__init__()

        self.modes = modes
        self.channels = channels
        self.svd_block_1 = SVD_Block_3(self.channels, self.channels, modes,modes, F_channels)
        self.svd_block_2 = SVD_Block_3(self.channels, self.channels, modes,modes, F_channels)
        self.W_FFT_out = FFT1D_block(self.channels, self.channels, modes,modes, pred_num)
    def forward(self, x):
        out_1 = self.svd_block_1(x)
        out_2 = self.svd_block_2(out_1)
        out = self.W_FFT_out(out_2)
        return out

