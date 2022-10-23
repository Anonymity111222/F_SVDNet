from sqlite3 import Time
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series
import torch
import time
import matplotlib
from pylab import xticks,yticks
#ETTH1
SCINet_MSE = [0.341,0.368,0.451,0.502,0.583 ]
F_SVD_MSE = [0.306,0.343,0.406,0.484,0.490]
SCINet_MAE = [0.379,0.395,0.457,0.497,0.560 ]
F_SVD_MAE = [0.349,0.372,0.423,0.484,0.504]
#ETTH2
# SCINet_MSE = [0.188,0.279,0.505,0.618,1.074 ]
# F_SVD_MSE = [0.173,0.244,0.397,0.478,0.726]
# SCINet_MAE = [0.288,0.358,0.504,0.560,0.761 ]
# F_SVD_MAE = [0.262,0.319,0.427,0.474,0.600]
#ETTm1
# SCINet_MSE = [0.126,0.169,0.191,0.365,0.713 ]
# F_SVD_MSE = [0.106,0.145,0.163,0.276,0.366]
# SCINet_MAE = [0.229,0.274,0.291,0.415,0.604]
# F_SVD_MAE = [0.203,0.241,0.252,0.340,0.394]
# x = [24,48,168,336,720]
# fig = plt.figure(dpi=300)
# plt.plot(x,SCINet_MSE,c='r',linestyle = '--',label='SCINet_MSE')
# plt.plot(x,F_SVD_MSE,c='orange',marker='+',linestyle = '--',label='F_SVD_MSE')
# plt.plot(x,SCINet_MAE,c='black',marker='o',linestyle = '-',label='SCINet_MAE')
# plt.plot(x,F_SVD_MAE,c='c',marker='*',linestyle = '-',label='F_SVD_MAE')
# fm=matplotlib.font_manager.FontProperties()
# fm.set_size(24)#设置字体为20，默认是10
# plt.xlabel('Future time steps',fontproperties=fm)
# plt.ylabel('Performances',fontproperties=fm)       
# plt.legend(loc='best', fontsize=15)
# #plt.legend(bbox_to_anchor=(0.42,0.99), fontsize=20)
# plt.tick_params(labelsize=24)
# matplotlib.rcParams.update({'font.size': 24})
# plt.tight_layout()
# plt.savefig('pred_figure/data_scales_F_SVD_ETTH1.png')


#ETTH1
# SCINet_MSE = [0.829,0.855,0.867,0.867 ]
# F_SVD_MSE = [0.573,0.754,0.778,0.814]
# SCINet_MAE = [0.704,0.717,0.728,0.728]
# F_SVD_MAE = [0.559,0.660,0.680,0.692]
#ETTH2
# SCINet_MSE = [3.083,2.993,2.200,4.632 ]
# F_SVD_MSE = [1.181,1.255,1.673,1.509]
# SCINet_MAE = [1.278,1.210,1.067,1.568 ]
# F_SVD_MAE = [0.756,0.789,0.902,0.860]
#ETTm1
# SCINet_MSE = [1.248,1.507,1.496,1.620 ]
# F_SVD_MSE = [0.451,0.512,0.572,0.829]
# SCINet_MAE = [0.833,0.933,0.931,0.973 ]
# F_SVD_MAE = [0.456,0.492,0.520,0.614]
# location = [1200,1440,1680,1920]
# x = range(len(location))
# fig = plt.figure(figsize=(6.5, 3*1.4),dpi=300)
# plt.bar([i-0.1 for i in x],SCINet_MSE,color='r',width=0.1,label='SCINet_MSE')
# plt.bar([i for i in x],F_SVD_MSE,color='orange',width=0.1,label='F_SVD_MSE')
# plt.bar([i+0.1 for i in x],SCINet_MAE,color='g',width=0.1,label='SCINet_MAE')
# plt.bar([i+0.2 for i in x],F_SVD_MAE,color='b',width=0.1,label='F_SVD_MAE')
# plt.xticks(x, location, fontsize=14)
# fm=matplotlib.font_manager.FontProperties()
# fm.set_size(24)#设置字体为20，默认是10
# plt.xlabel('Future time steps',fontproperties=fm)
# plt.ylabel('Error',fontproperties=fm)       
# plt.legend(ncol=2,bbox_to_anchor=(0,1.0,1,0), fontsize=15)
# plt.tick_params(labelsize=24)
# matplotlib.rcParams.update({'font.size': 24})
# yticks(np.linspace(0.00,2.50,3,endpoint=True))
# plt.tight_layout()
# plt.savefig('pred_figure/data_scales_F_SVD_ETTm1_erro.png')


#24========================================================================================================================
# true_data = np.load('exp/ett_results/FFT-1_ETTh1_ftM_sl96_ll24_pl24_lr0.0001_bs32/true_scales.npy')
# FFT_1 = np.load('exp/ett_results/FFT-1_ETTh1_ftM_sl96_ll24_pl24_lr0.0001_bs32/pred_scales.npy')
# FFT_1_ACT = np.load('exp/ett_results/FFT-1-actFFT-1-act_ETTh1_ftM_sl24_ll24_pl24_lr0.0001_bs32/pred_scales.npy')
# FFT_4 = np.load('exp/ett_results/FFT-4_ETTh1_ftM_sl96_ll24_pl24_lr0.0001_bs32/pred_scales.npy')
# FFT_4_ACT = np.load('exp/ett_results/FFT-4-actFFT-4-act_ETTh1_ftM_sl96_ll24_pl24_lr0.0001_bs32/pred_scales.npy')
# SCINet = np.load('/home/weiwang/timeseries/TimeSeries/SCINet-main/exp/ett_results/SCINet_ETTh1_ftM_sl96_ll24_pl24_lr0.0001_bs32_hid1_s1_l3_dp0.5_invFalse_itr0/pred_scales.npy')
# x=[i for i in range(24)]

# num_attribute = 6
# num_seq = 500
# fig = plt.figure(dpi=300)
# plt.plot(x,true_data[num_seq][:,num_attribute],c='r',label='Groud true')
# plt.plot(x,FFT_1[num_seq][:,num_attribute],c='orange',marker='+',label='FFT-1')
# plt.plot(x,FFT_4[num_seq][:,num_attribute],c='olive',marker='o',label='FFT-4')
# plt.plot(x,FFT_1_ACT[num_seq][:,num_attribute],c='c',marker='*',label='FFT-1 with LeakyRelu')
# plt.plot(x,FFT_4_ACT[num_seq][:,num_attribute],c='teal',marker='<',label='FFT-4 with LeakyRelu')
# plt.plot(x,SCINet[num_seq][:,num_attribute],c='black',marker='v',label='SCINet')
# plt.xlabel('Future time steps')
# plt.ylabel('Prediction results')       
# plt.legend(loc='best')
# plt.tight_layout()
# plt.savefig('pred_figure/data_scales_500_24_6.png')


#48========================================================================================================================
# true_data = np.load('exp/ett_results/FFT-SVDFFT-SVD_ETTh1_ftM_sl96_ll48_pl48_lr0.0001_bs32/true_scales.npy')
# F_SVD = np.load('exp/ett_results/FFT-SVDFFT-SVD_ETTh1_ftM_sl96_ll48_pl48_lr0.0001_bs32/pred_scales.npy')
# SCINet = np.load('/home/weiwang/timeseries/TimeSeries/SCINet-main/exp/ett_results/SCINet_ETTh1_ftM_sl96_ll48_pl48_lr0.0001_bs32_hid1_s1_l3_dp0.5_invFalse_itr0/pred_scales.npy')

# x=[i for i in range(48)]

# num_attribute = 0
# num_seq = 1000
# fig = plt.figure(dpi=300)
# plt.plot(x,true_data[num_seq,:,num_attribute],c='r',label='Groud true')
# plt.plot(x,F_SVD[num_seq,:,num_attribute],c='orange',marker='+',label='F-SVD')
# plt.plot(x,SCINet[num_seq,:,num_attribute],c='black',marker='v',label='SCINet')
# fm=matplotlib.font_manager.FontProperties()
# fm.set_size(24)#设置字体为20，默认是10
# plt.xlabel('Future time steps',fontproperties=fm)
# plt.ylabel('Prediction results',fontproperties=fm)       
# plt.legend(ncol=2,loc='best', fontsize=15)
# #plt.legend(ncol=2,bbox_to_anchor=(0,1.1,1,0.2), fontsize=15)
# plt.tick_params(labelsize=24)
# matplotlib.rcParams.update({'font.size': 24})
# plt.tight_layout()
# yticks(np.linspace(-15,12,3,endpoint=True))
# plt.savefig('pred_figure/data_scales_F_SVD_48_1000_0.png')

#168=====================================================================================================================

# true_data = np.load('exp/ett_results/FFT-1_ETTh1_ftM_sl720_ll168_pl168_lr0.0001_bs32/true_scales.npy')
# FFT_1 = np.load('exp/ett_results/FFT-1_ETTh1_ftM_sl720_ll168_pl168_lr0.0001_bs32/pred_scales.npy')
# FFT_1_ACT = np.load('exp/ett_results/FFT-1-actFFT-1-act_ETTh1_ftM_sl720_ll168_pl168_lr0.0001_bs32/pred_scales.npy')
# FFT_4 = np.load('exp/ett_results/FFT-4_ETTh1_ftM_sl720_ll168_pl168_lr0.0001_bs32/pred_scales.npy')
# FFT_4_ACT = np.load('exp/ett_results/FFT-4-actFFT-4-act_ETTh1_ftM_sl720_ll168_pl168_lr0.0001_bs32/pred_scales.npy')
# SCINet = np.load('/home/weiwang/timeseries/TimeSeries/SCINet-main/exp/ett_results/SCINet_ETTh1_ftM_sl720_ll168_pl168_lr0.0001_bs32_hid1_s1_l3_dp0.5_invFalse_itr0/pred_scales.npy')

# x=[i for i in range(168)]


# fig = plt.figure(dpi=300,figsize=(50, 5))
# plt.plot(x,true_data[0][:,6],c='r',label='Groud true')
# plt.plot(x,FFT_1[0][:,6],c='orange',marker='+',label='FFT-1')
# #plt.plot(x,FFT_4[0][:,6],c='olive',marker='o',label='FFT-4')
# plt.plot(x,FFT_1_ACT[0][:,6],c='c',marker='*',label='FFT-1 with LeakyRelu')
# #plt.plot(x,FFT_4_ACT[0][:,6],c='teal',marker='<',label='FFT-4 with LeakyRelu')
# plt.plot(x,SCINet[0][:,6],c='black',marker='v',label='SCINet')
# plt.xlabel('Future time steps')
# plt.ylabel('Prediction results')       
# plt.legend(loc='best')
# plt.tight_layout()
# plt.savefig('pred_figure/data_scales_168_6.png')

# file_path = 'exp/ett_results/Taley_ETTh1_ftM_sl96_ll24_pl24_lr0.0001_bs32_hid1_s1_l3_dp0.5_invFalse_itr0/kernel3/metrics.npy'
# data = np.load(file_path)
# print(data)

# a = np.random.rand(128)
# b = np.fft.fft(a)
# n = b.shape
# x = [i for i in range(n[0])]
# plt.plot(x,b.real)
# plt.plot(x,b.imag)
# plt.savefig('pred_figure/data_fft_1.png')

# def naive_arg_topK(matrix, K, axis=0):
#     """
#     perform topK based on np.argsort
#     :param matrix: to be sorted
#     :param K: select and sort the top K items
#     :param axis: dimension to be sorted.
#     :return:
#     """
#     full_sort = np.argsort(matrix, axis=axis)
#     return full_sort.take(np.arange(K), axis=axis)
# from scipy.fftpack import fft,ifft
# a = np.random.rand(128)
# x = [i for i in range(128)]
# xx = [i for i in range(100)]
# b = fft(a)

# bb = np.abs(b)
# absbb = naive_arg_topK(bb,10)
# mink = bb[absbb[-1]]
# d = bb>mink
# b = b*d

# x = [i for i in range(128)]
# xx = [i for i in range(128)]

# a = torch.rand(128)
# data = torch.linspace(1, 10, steps=128)
# y = torch.sin(data)
# a = y+a 
# b = torch.fft.rfft(a)

# bb = torch.abs(b)
# absb = torch.topk(bb,k=10,largest=False)
# max_value = absb.values[-1]
# tf = bb>max_value
# b = b*tf
# c = torch.fft.irfft(b,n=128)

# plt.plot(x,a.numpy())
# plt.plot(xx,c.numpy())
# # plt.tight_layout()
# plt.savefig('predfigures/fft.png')


