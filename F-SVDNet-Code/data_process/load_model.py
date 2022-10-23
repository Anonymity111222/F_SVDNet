import torch

from torchvision import utils as vutils
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


model2 = torch.load('exp/ETT_checkpoints/FFT-SVD_ETTh1_ftM_sl96_ll24_pl24_lr0.0001_bs32/ETTh124.bin')
model1 = torch.load('exp/ETT_checkpoints/FFT-SVD_ETTh2_ftM_sl96_ll24_pl24_lr0.0001_bs32/ETTh224.bin')
weight1 = model2 ['model']['svd_block_1.weight_SVD'].cpu()
weight2 = model1['model']['svd_block_1.weight_SVD'].cpu()

# plot=sns.heatmap(act_weight) 
# plt.savefig('pred_figure/data_act_weight_imag.png')
# u1,s1,v1 = torch.svd(weight1)
# u2,s2,v2 = torch.svd(weight2)
# print(s1)
# print(s2)
# print(s1-s2)
plot=sns.heatmap((weight1-weight2).numpy()) 
plt.savefig('pred_figure/data_weight.png')