import torch
import torch.nn as nn
import math

'''

SqueezeNet with Simple Bypass -- Residual Connections after every other Fire Module

'''

class SqueezeNet(nn.Module):

    '''

    base_e: num of filters in the expansion layer for the first fire module
    incr_e: increment value for filters in subsequent fire modules to the first
    pct_3x3: % value of 3x3 filters in the 3x3 expansion covnolution, relative to the entire expansion layer.
    freq: frequency of fire modules at which the incr_e is applied
    sr: squeeze ratio, denoting the amount of 1x1 filter in the squeeze layer of the fire module relative to the total set of expansion filters.

    '''

    def __init__(self, base_e, incr_e, pct_3x3, freq, sr, f_modules = 8):

        super().__init__()

        e = [ ]

        for f in range(f_modules):

            e.append(base_e + (incr_e * math.floor(f / freq)))


        self.conv1 = BasicConv2d(

            in_channels = 3,
            out_channels = 96,
            kernel_size = 7, 
            stride = 2,
            padding = 2

        )    

        self.maxpool1 = nn.MaxPool2d(

            kernel_size = 3,
            stride = 2

        )

        self.fire2 = FireModule(

            in_channels = 96,
            s_1x1 = int(sr * e[0]),
            e_1x1 = int(e[0] * (1 - pct_3x3)),
            e_3x3 = int(e[0] * pct_3x3)

        )

        self.fire3 = FireModule(
        
            in_channels = int(e[0] * ( 1 - pct_3x3)) + int(e[0] * pct_3x3),
            s_1x1 = int(sr * e[1]),
            e_1x1 = int(e[1] * (1 - pct_3x3)),
            e_3x3 = int(e[1] * pct_3x3),
            residual = True

        )

        self.fire4 = FireModule(

            in_channels = int(e[1] * ( 1 - pct_3x3)) + int(e[1] * pct_3x3),
            s_1x1 = int(sr * e[2]),
            e_1x1 = int(e[2] * (1 - pct_3x3)),
            e_3x3 = int(e[2] * pct_3x3)

        )

        self.maxpool4 = nn.MaxPool2d(

            kernel_size = 3,
            stride = 2

        )

        self.fire5 = FireModule(

            in_channels = int(e[2] * ( 1 - pct_3x3)) + int(e[2] * pct_3x3),
            s_1x1 = int(sr * e[3]),
            e_1x1 = int(e[3] * (1 - pct_3x3)),
            e_3x3 = int(e[3] * pct_3x3),
            residual = True
        )

        self.fire6 = FireModule(

            in_channels = int(e[3] * ( 1 - pct_3x3)) + int(e[3] * pct_3x3),
            s_1x1 = int(sr * e[4]),
            e_1x1 = int(e[4] * (1 - pct_3x3)),
            e_3x3 = int(e[4] * pct_3x3)

        )

        self.fire7 = FireModule(

            in_channels = int(e[4] * ( 1 - pct_3x3)) + int(e[4] * pct_3x3),
            s_1x1 = int(sr * e[5]),
            e_1x1 = int(e[5] * (1 - pct_3x3)),
            e_3x3 = int(e[5] * pct_3x3),
            residual = True 

        )

        self.fire8 = FireModule(

            in_channels = int(e[5] * ( 1 - pct_3x3)) + int(e[5] * pct_3x3),
            s_1x1 = int(sr * e[6]),
            e_1x1 = int(e[6] * (1 - pct_3x3)),
            e_3x3 = int(e[6] * pct_3x3)

        )

        self.maxpool8 = nn.MaxPool2d(

            kernel_size = 3,
            stride = 2

        )

        self.fire9 = FireModule(

            in_channels = int(e[6] * ( 1 - pct_3x3)) + int(e[6] * pct_3x3),
            s_1x1 = int(sr * e[6]),
            e_1x1 = int(e[7] * (1 - pct_3x3)),
            e_3x3 = int(e[7] * pct_3x3),
            residual = True

        )

        self.conv10 = BasicConv2d(

            in_channels = int(e[7] * ( 1 - pct_3x3)) + int(e[7] * pct_3x3),
            out_channels = 1000,
            kernel_size = 1

        )

        self.avgpool10 = nn.AdaptiveAvgPool2d(

            output_size = (1, 1)

        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.maxpool4(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.maxpool8(x)
        x = self.fire9(x)
        x = self.conv10(x)
        x = self.avgpool10(x)

        return x

class FireModule(nn.Module):

    def __init__(

        self, 
        in_channels:int, 
        s_1x1:int,
        e_1x1:int,
        e_3x3:int,
        residual:bool = False

    ):

        self.residual = residual

        super().__init__()
        
        self.squeeze_1x1 = BasicConv2d(

            in_channels = in_channels,
            out_channels = s_1x1,
            kernel_size = 1,

        )
        
        
        self.expand_1x1 = BasicConv2d(

            in_channels = s_1x1,
            out_channels = e_1x1,
            kernel_size = 1

        )

        self.expand_3x3 = BasicConv2d(

            in_channels = s_1x1,
            out_channels = e_3x3,
            kernel_size = 3,
            padding = 1

        )


    def forward(self, x):

        x_squeeze = self.squeeze_1x1(x)
        x_expand_1 = self.expand_1x1(x_squeeze)
        x_expand_2 = self.expand_3x3(x_squeeze)

        x_out = torch.cat(

                [x_expand_1, x_expand_2],
                dim = 1,

        )

        if self.residual:
       
            x_out += x

        return x_out

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, norm = False):

        self.norm = norm

        super().__init__()

        self.conv = nn.Conv2d(

            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding

        )

        self.relu = nn.ReLU()

        if self.norm:

            self.batchnorm = nn.BatchNorm2d(

                num_features = out_channels 

            )

    def forward(self, x):

        x = self.conv(x)

        if self.norm:
            x = self.batchnorm(x)

        x = self.relu(x)

        return x
