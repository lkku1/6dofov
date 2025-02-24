import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules.spectral_norm import spectral_norm as _spectral_norm

class ZeroPad(nn.Module):
    def __init__(self, padding):
        super(ZeroPad, self).__init__()
        if isinstance(padding, int):
            self.h = padding
            self.w = padding
        else:
            self.h = padding[0]
            self.w = padding[1]

    def forward(self, x):
        x = F.pad(x, (self.w, self.w, self.h, self.h))
        return x

class ZeroPad3d(nn.Module):
    def __init__(self, padding):
        super(ZeroPad3d, self).__init__()
        if isinstance(padding, int):
            self.t = padding
            self.h = padding
            self.w = padding
        else:
            self.t = padding[0]
            self.h = padding[1]
            self.w = padding[2]

    def forward(self, x):
        x = F.pad(x, (self.w, self.w, self.h, self.h, self.t, self.t))
        return x

class LRPad(nn.Module):
    def __init__(self,padding):
        super().__init__()
        if isinstance(padding,int):
            self.h = padding
            self.w = padding
        else:
            self.h = padding[0]
            self.w = padding[1]

    def forward(self,x):
        _,_,H,W = x.shape
        assert H%2==0 and W%2==0
        if self.h==0 and self.w==0:
            return x
        return F.pad(F.pad(x,pad=(self.w,self.w,0,0),mode='circular'),pad=(0,0,self.h,self.h))

class LRPad3d(nn.Module):
    def __init__(self,padding):
        super().__init__()
        if isinstance(padding,int):
            self.t = padding
            self.h = padding
            self.w = padding
        else:
            self.t = padding[0]
            self.h = padding[1]
            self.w = padding[2]

    def forward(self,x):
        _, _, T, H,W = x.shape
        if self.h==0 and self.w==0:
            return x

        return F.pad(F.pad(x,pad=(self.w,self.w,0,0, 0, 0),mode='circular'),pad=(0,0,self.h,self.h, self.t, self.t))

def _make_pad(padding=0, pad='zeropad'):
    if pad == 'lrpad':
        return LRPad(padding)
    elif pad == 'zeropad':
        return ZeroPad(padding)

def _make_pad3d(padding=0, pad='zeropad'):
    if pad == 'lrpad':
        return LRPad3d(padding)
    elif pad == 'zeropad':
        return ZeroPad3d(padding)


class PadConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,dilation=1,bias=True,groups=1):
        super().__init__()
        if isinstance(dilation,int):
            dh = dilation
            dw = dilation
        else:
            dh = dilation[0]
            dw = dilation[1]
        if isinstance(kernel_size,int):
            h = (kernel_size // 2)*dh
            w = (kernel_size // 2)*dw
        else:
            h = (kernel_size[0] // 2)*dh
            w = (kernel_size[1] // 2)*dw
        self.pad = _make_pad([h,w])
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,dilation=dilation,bias=bias,groups=groups)
    def forward(self,x):
        return self.conv(self.pad(x))

class PadConv3d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,dilation=1,bias=True,groups=1):
        super().__init__()
        if isinstance(dilation,int):
            dt = dilation
            dh = dilation
            dw = dilation
        else:
            dt = dilation[0]
            dh = dilation[1]
            dw = dilation[2]
        if isinstance(kernel_size,int):
            t = (kernel_size // 2)*dt
            h = (kernel_size // 2)*dh
            w = (kernel_size // 2)*dw
        else:
            t = (kernel_size[0] // 2)*dt
            h = (kernel_size[1] // 2)*dh
            w = (kernel_size[2] // 2)*dw
        self.pad = _make_pad3d([t, h, w])
        self.conv = spectral_norm(nn.Conv3d(in_channels,out_channels,kernel_size,stride=stride,dilation=dilation,bias=bias,groups=groups), not bias)
    def forward(self,x):

        return self.conv(self.pad(x))

def spectral_norm(module, mode=True):
    if mode:
        return _spectral_norm(module)
    return module