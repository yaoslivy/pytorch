
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter




if __name__ == "__main__": 
    input_feat = torch.tensor([[4, 1, 7, 5], [4, 4, 2, 5], [7, 7, 2, 4], [1, 0, 2, 4]], dtype=torch.float32)
    print(input_feat)
    print(input_feat.shape)
    # in_channels=1, out_channels=1 
    conv2d = nn.Conv2d(1, 1, (2, 2), stride=1, padding='same', bias=True)
    print(conv2d.weight)
    print(conv2d.bias)
    
    conv2d = nn.Conv2d(1, 1, (2, 2), stride=1, padding='same', bias=False)
    kernels = torch.tensor([[[[1, 0], [2, 1]]]], dtype=torch.float32)
    conv2d.weight = Parameter(kernels, requires_grad=False)
    print(conv2d.weight)
    print(conv2d.bias)


    # DW conv
    # data shape : 3*5*5
    x = torch.rand((3, 5, 5)).unsqueeze(0)
    print(x.shape)
    # DW conv: in_channels_dw == out_channels_dw
    in_channels_dw = x.shape[1]
    out_channels_dw = x.shape[1]
    kernel_size = 3
    stride = 1
    # DW conv: groups shape is in_channels_dw
    dw = nn.Conv2d(in_channels_dw, out_channels_dw, kernel_size, stride, groups=in_channels_dw)
    # PW conv 
    in_channels_pw = out_channels_dw
    out_channels_pw = 4
    kernel_size_pw = 1
    pw = nn.Conv2d(in_channels_pw, out_channels_pw, kernel_size_pw, stride)
    out = pw(dw(x))
    print(out.shape)
