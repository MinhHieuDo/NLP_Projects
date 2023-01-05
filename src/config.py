import os
import torch
# from typing import List
# from sys import platform

# Folders
project_name = 'sinkhorn-rom'
project_dir  = os.path.dirname(os.path.dirname(__file__))
result_dir  = project_dir + '/result/'
data_dir     = project_dir + '/data/'
fit_dir      = project_dir + '/fit/'
print('result_dir',result_dir)
print('data_dir',data_dir)
print('fit_dir',fit_dir)

# For pytorch: get device and set dtype
use_cuda = torch.cuda.is_available()
device = torch.device('cuda') if use_cuda else torch.device('cpu')
dtype = torch.double # Autograd does not work so far with torch.double
