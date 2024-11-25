import torch
import torch.nn as nn
from ops import BasicConv2d

# src -- https://arxiv.org/pdf/1905.02244

class MobileBlock(nn.Module):

    '''

    Inverted Residual Block with integrated Squeeze and Excitation, and a Depthwise Convolution

    expansion_size: expansion size for the first 1x1 conv.

    '''

    def __init__(

            
        self,
        in_channels,
        out_channels,
        kernel_size,
        expansion_ch,
        stride = 1,
        padding = 0,
        residual = True,
        squeeze_excite = True,
        non_linearity = 'h_swish'
            
    ):


        super().__init__()

        self.residual = residual
        self.squeeze_excite = squeeze_excite

        non_linearity = non_linearity.lower()

        if non_linearity == 'h_swish':

            self.act = nn.Hardswish()
            se_non_linearity = 'h_sigmoid'

        elif non_linearity == 'relu':

            self.act = nn.ReLU()
            se_non_linearity = 'relu'

        self.conv_exp = BasicConv2d(

            in_channels = in_channels,
            out_channels = expansion_ch,
            kernel_size = 1,

        )

        self.conv_dw = BasicConv2d(

            in_channels = expansion_ch,
            out_channels = expansion_ch,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            groups = expansion_ch

        )

        if self.squeeze_excite:

            self.se = SEBlock(

                in_channels = expansion_ch,
                reduct_ratio = .25, # as specified in the paper.
                non_linearity = se_non_linearity 

            )

        self.conv_proj = BasicConv2d(

            in_channels = expansion_ch,
            out_channels = out_channels,
            kernel_size = 1

        )

        if self.residual:

            self.identity = nn.Identity()


    def forward(self, x):

        x_out = self.conv_exp(x)
        x_out = self.conv_dw(x_out)

        if self.squeeze_excite:

            x_out = self.se(x_out)
   
        x_out = self.conv_proj(x_out)
 
        if self.residual:

            x_residual = self.identity(x)
            x_out += x_residual
        
            return x_out

        return x_out

class SEBlock(nn.Module):

    '''

    Squeeze and Excitation Block

    '''

    def __init__(

            self, 
            in_channels, 
            reduct_ratio,
            non_linearity = 'h_sigmoid'
    ):

        reduct_ch = int(in_channels * reduct_ratio)

        self.non_linearity = non_linearity.lower()

        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(

            output_size = (1, 1)

        )

        self.fc1 = nn.Linear(

            in_features = in_channels,
            out_features = reduct_ch

        )
       
        if self.non_linearity == 'relu':

            self.act1 = nn.ReLU()

        elif self.non_linearity == 'h_sigmoid':

            self.act1 = nn.Hardswish()

        self.fc2 = nn.Linear(

            in_features = reduct_ch,
            out_features = in_channels

        )

        self.act2 = nn.Hardsigmoid()

    def forward(self, x):

        x_scale = self.avgpool(x)

        x_scale = torch.reshape(

            x_scale,
            shape = (

                x_scale.size( dim = 0),
                x_scale.size(dim = 1)

            )
        )


        x_scale = self.fc1(x_scale)
        x_scale = self.act1(x_scale)
        x_scale = self.fc2(x_scale)
        x_scale = self.act2(x_scale)

        x_scale = torch.reshape(

            x_scale,
            shape = (

                x_scale.size(dim = 0), 
                x_scale.size(dim = 1),
                1,
                1

            )
        )

        x *= x_scale

        return x
