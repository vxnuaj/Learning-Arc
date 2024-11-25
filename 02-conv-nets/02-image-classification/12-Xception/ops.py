import torch.nn as nn


class EBlock(nn.Module):

    def __init__(

        self,
        in_channels,
        out_channels,
        init = False

    ):

        self.init = init

        if isinstance(out_channels, int):
            out_channels = [out_channels for _ in range(2)]

        super().__init__()

        self.relu = nn.ReLU()

        self.sep_conv1 = SeperableConv2d(
            
            in_channels = in_channels,
            out_channels = out_channels[0]

        )

        self.sep_conv2 = SeperableConv2d(

            in_channels = out_channels[0],
            out_channels = out_channels[1]

        )

        self.maxpool = nn.MaxPool2d(

            kernel_size = 3,
            stride = 2,
            padding = 1

        )

        self.res_cnnct = BasicConv2d(

            in_channels = in_channels,
            out_channels = out_channels[1],
            kernel_size = 1,
            stride = 2,
            act = False

        )

        return


    def forward(self, x):

        if self.init == False:
            x_out = self.relu(x) 
        else:
            x_out = x

        x_out = self.sep_conv1(x_out)
        x_out = self.relu(x_out)
        x_out = self.sep_conv2(x_out)
        x_out = self.maxpool(x_out)

        x_res = self.res_cnnct(x)

        x_out += x_res

        return x_out

class MBlock(nn.Module):

    def __init__(

        self,
        in_channels,
        out_channels

        ):

        super().__init__()

        self.relu = nn.ReLU()
        self.res_cnnct = nn.Identity() # identity skip connection 

        self.sep_conv1 = SeperableConv2d(

            in_channels = in_channels,
            out_channels = out_channels,

        )

        self.sep_conv2 = SeperableConv2d(

            in_channels = in_channels,
            out_channels = out_channels
        
        )

        self.sep_conv3 = SeperableConv2d(

            in_channels = in_channels,
            out_channels = out_channels

        )


    def forward(self, x):

        x_out = self.relu(x)
        x_out = self.sep_conv1(x_out)
        x_out = self.relu(x_out)
        x_out = self.sep_conv2(x_out)
        x_out = self.relu(x_out)
        x_out = self.sep_conv3(x_out)
        
        x_res = self.res_cnnct(x) # residual identity connection -- vxnuaj.com/blog/residuals

        x_out += x_res

        return x_out


class SeperableConv2d(nn.Module):

    def __init__(

        self,
        in_channels,
        out_channels,
        act = False

    ):

        self.act = act

        super().__init__()

        self.conv_1x1 = BasicConv2d(

            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 1,
            act = False

        )

        self.conv_3x3 = BasicConv2d(

            in_channels = out_channels,
            out_channels = out_channels,
            kernel_size = 3,
            groups = out_channels,
            padding = 1,
            act = False

        )

        if self.act:

            self.relu = nn.ReLU()

    def forward(self, x):

        x = self.conv_1x1(x)
        x = self.conv_3x3(x)

        if self.act:

            x = self.relu(x)

        return x


class BasicConv2d(nn.Module):

    def __init__(
        
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride = 1,
        padding = 0,
        groups = 1,
        act = True

    ):
        
        self.act = act

        super().__init__()

        self.conv = nn.Conv2d(

            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            groups = groups

        )

        self.batchnorm = nn.BatchNorm2d(

            num_features = out_channels 

        )

        if self.act:
            self.relu = nn.ReLU()

    def forward(self, x):

        x = self.conv(x)
        x = self.batchnorm(x)

        if self.act:
            x = self.relu(x)

        return x
