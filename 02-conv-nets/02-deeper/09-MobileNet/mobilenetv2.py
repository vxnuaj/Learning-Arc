import torch.nn as nn
import torchvision.transforms.v2 as v2

class MobileNetV2(nn.Module):

    '''
    alpha: width multiplier
    rho: resolution multiplier
    classes: total num of classes in the dataset
    '''

    def __init__(self, alpha = 1, rho = 1, classes = 1000):

        assert 0 < alpha <=1, ValueError("alpha must be in range 0 < alpha <= 1")
        assert 0 < rho <=1, ValueError("rho must be in range 0 < rho <= 1")
        assert alpha is not None, ValueError("alpha must not be None")
        assert rho is not None, ValueError("rho must not be None")

        super().__init__()

        target_size = int(224 * rho)

        self.transforms = v2.Compose([
                v2.Resize([target_size, target_size]),
                v2.Normalize(
                    mean = [.485, .456, .406],
                    std = [.229, .224, .225]
                )
        ])

        self.conv_3x3 = BasicConv2d(

            in_channels = 3,
            out_channels = 32,
            kernel_size = 3,
            stride = 2,
            padding = 1

        )

        self.conv_blk_1 = InvertedResidual(

            in_channels = 32,
            out_channels = 16,
            alpha = alpha,
            t = 1,
            alt_ch = True

        )

        self.conv_blk_2_3 = nn.Sequential( 

            InvertedResidual(

                in_channels = 16,
                out_channels = 24,
                alpha = alpha,
                t = 6,
                reduct_hw = True,

            ),

            InvertedResidual(

                in_channels = 24,
                out_channels = 24,
                alpha = alpha,
                t = 6,

            )

        )

        self.conv_blk_4_6 = nn.Sequential(

            InvertedResidual(

                in_channels = 24,
                out_channels = 32,
                alpha = alpha,
                t = 6,
                alt_ch = True,
                reduct_hw = True,

            ),

            InvertedResidual(
                
                in_channels = 32,
                out_channels = 32,
                alpha = alpha,
                t = 6

            ),

            InvertedResidual(

                in_channels = 32,
                out_channels = 32,
                alpha = alpha,
                t = 6

            )
    
        )


        self.conv_blk_7_10 = nn.Sequential(

            InvertedResidual(

                in_channels = 32,
                out_channels = 64,
                alpha = alpha,
                t = 6,
                alt_ch = True,
                reduct_hw = True

            ),

            InvertedResidual(

                in_channels = 64,
                out_channels = 64,
                alpha = alpha,
                t = 6

            ),

            InvertedResidual(

                in_channels = 64,
                out_channels = 64,
                alpha = alpha, 
                t = 6

            ),

            InvertedResidual(

                in_channels = 64,
                out_channels = 64,
                alpha = alpha, 
                t = 6

            )
        )

        self.conv_blk_11_13 = nn.Sequential(

            InvertedResidual(

                in_channels = 64,
                out_channels = 96,
                alpha = alpha,
                t = 6,
                alt_ch = True

            ),

            InvertedResidual(

                in_channels = 96,
                out_channels = 96,
                alpha = alpha,
                t = 6

            ),

            InvertedResidual(

                in_channels = 96,
                out_channels = 96,
                alpha = alpha,
                t = 6

            )

        )

        self.conv_blk_14_16 = nn.Sequential(


            InvertedResidual(

                in_channels = 96,
                out_channels = 160,
                alpha = alpha,
                t = 6,
                alt_ch = True,
                reduct_hw = True,

                ),

            InvertedResidual(

                in_channels = 160,
                out_channels = 160,
                alpha = alpha,
                t = 6
            
            ),

            InvertedResidual(

                in_channels = 160,
                out_channels = 160,
                alpha = alpha,
                t = 6
            
            )

        )

        self.conv_blk_17 = InvertedResidual(

            in_channels = 160,
            out_channels = 320,
            alpha = alpha,
            t = 6,
            alt_ch = True,

        )

        self.conv_1x1 = BasicConv2d(

            in_channels = 320,
            out_channels = 1280,
            kernel_size = 1,

        )

        self.avgpool = nn.AdaptiveAvgPool2d(

            output_size = (1, 1)

        )

        self.conv_1x1_out = BasicConv2d(

            in_channels = 1280,
            out_channels = classes,
            kernel_size = 1

        )


    def forward(self, x):

        x = self.transforms(x)
        x  = self.conv_3x3(x)
        x = self.conv_blk_1(x)
        x = self.conv_blk_2_3(x)
        x = self.conv_blk_4_6(x)
        x = self.conv_blk_7_10(x)
        x = self.conv_blk_11_13(x)
        x = self.conv_blk_14_16(x)
        x = self.conv_blk_17(x)
        x = self.conv_1x1(x)
        x = self.avgpool(x)
        x = self.conv_1x1_out(x)

        return x


class InvertedResidual(nn.Module):

    '''

    alpha: width mul, where 0 < alpha <= 1
    t: expansion ratio

    '''

    def __init__(
        
        self, 
        in_channels:int, 
        out_channels:int, 
        alpha = 1, 
        t = None, 
        alt_ch = False, 
        reduct_hw = False
        
        ):

        assert t is not None, "expansion factor, t, must not be None"
        assert alpha is not None, "width multiplier, alpha, must noe be None"

        super().__init__()

        self._alt_ch = alt_ch
        self._reduct_hw = reduct_hw

        in_channels = int(in_channels * alpha)
        expanded_channels = int(in_channels * t)
        out_channels = int(out_channels * alpha)

        padding = 1

        if self._reduct_hw and self._alt_ch or self._reduct_hw:

            stride = 2

            self.residual_connection = BasicConv2d(

                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = stride
            
            )

        elif self._alt_ch:

            stride = 1

            self.residual_connection = BasicConv2d(

                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = stride
            )

        else:

            stride = 1

            self.residual_connection = nn.Identity()


        self.conv_1x1_in = BasicConv2d(
            
            in_channels = in_channels, 
            out_channels = expanded_channels,
            kernel_size = 1,

        )

        self.conv_3x3 = BasicConv2d(

            in_channels = expanded_channels,
            out_channels = expanded_channels,
            kernel_size = 3,
            stride = stride,
            padding = padding,
            groups = expanded_channels

        )

        self.conv_1x1_out = BasicConv2d(

            in_channels = expanded_channels,
            out_channels = out_channels,
            kernel_size = 1,
            out = True

        )

    def forward(self, x):

        x_res = self.conv_1x1_in(x)
        x_res = self.conv_3x3(x_res)
        x_res = self.conv_1x1_out(x_res)

        x_res += self.residual_connection(x) # residual connection

        return x_res

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 0, groups = 1, alpha = 1, out = False):

        super().__init__()


        self.out = out

        self.conv = nn.Conv2d(
            
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            groups = groups

        )

        self.bn = nn.BatchNorm2d(num_features = out_channels)
        self.relu = nn.ReLU6()

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)

        if self.out is True:
            return x

        x = self.relu(x)

        return x
