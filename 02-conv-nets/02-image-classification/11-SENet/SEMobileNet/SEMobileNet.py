import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
from SEBlock import SEBlock

class SEMobileNetV1(nn.Module):

    '''

    alpha: width multiplier
    rho: resolution multiplier

    '''

    def __init__(self, alpha:float = 1, rho:float = 1, reduct_ratio = 16):

        assert 0 < alpha <=1, ValueError("alpha must be in range 0 < alpha <= 1")
        assert 0 < rho <=1, ValueError("rho must be in range 0 < rho <= 1")

        super().__init__()

        target_size = int(224 * rho)

        self.transforms = v2.Compose([
                v2.Resize([target_size, target_size]),
                v2.Normalize(
                    mean = [.485, .456, .406],
                    std = [.229, .224, .225]
                )
        ])

        self.conv_1_in = BasicConv2d(

            in_channels = 3,
            out_channels = 32,
            kernel_size = 3,
            stride = 2,
            padding = 1,
            alpha = alpha

        )

        self.conv_2_3_dw = nn.Sequential(
            
            DepthSepConv2d(

                in_channels = 32,
                out_channels = (32, 64),
                stride = (1, 1),
                padding = (1, 0),
                alpha = alpha

            ),
            
            SEBlock(
                
                channels = 64,
                reduct_ratio = reduct_ratio,
                alpha = alpha
                
            )
            
        )


        self.conv_4_5_dw = nn.Sequential(
            
            DepthSepConv2d(

                in_channels = 64,
                out_channels = (64, 128),
                stride = (2, 1),
                padding = (1, 0),
                alpha = alpha

            ),
            
            SEBlock(
                
                channels = 128,
                reduct_ratio = reduct_ratio,
                alpha = alpha
                
            )
        )

        self.conv_6_7_dw = nn.Sequential(
        
            DepthSepConv2d(

                in_channels = 128,
                out_channels = (128, 128),
                stride = (1, 1),
                padding = (1, 0),
                alpha = alpha

            ),
            
            SEBlock(
                
                channels = 128,
                reduct_ratio = reduct_ratio,
                alpha = alpha
                
            )

        )

        self.conv_8_9_dw = nn.Sequential(
            
            DepthSepConv2d(

                in_channels = 128,
                out_channels = (128, 256),
                stride = (2, 1),
                padding = (1, 0),
                alpha = alpha

            ),
            
            SEBlock(
                
                channels = 256,
                reduct_ratio = reduct_ratio,
                alpha = alpha
                
            )

        )

        self.conv_10_11_dw = nn.Sequential( 
                                           
            DepthSepConv2d(

                in_channels = 256,
                out_channels = (256, 256),
                stride = (1, 1),
                padding = (1, 0),
                alpha = alpha

            ), 
            
            SEBlock(
                
                channels = 256,
                reduct_ratio = reduct_ratio,
                alpha = alpha
            )
            
        )

        self.conv_12_13_dw = nn.Sequential(
        
            DepthSepConv2d(

                in_channels = 256,
                out_channels = (256, 512),
                stride = (2, 1),
                padding = (1, 0),
                alpha = alpha

            ),
            
            SEBlock(
                
                channels = 512,
                reduct_ratio = reduct_ratio,
                alpha = alpha
            )
            
        )

        self.conv_14_23_dw = nn.Sequential(
            *[
                nn.Sequential(
                    DepthSepConv2d(
                        in_channels=512,
                        out_channels=(512, 512),
                        stride=(1, 1),
                        padding=(1, 0),
                        alpha=alpha
                    ),
                    SEBlock(
                        channels=512,
                        reduct_ratio=reduct_ratio,
                        alpha = alpha
                    )
                )
                for _ in range(5)
            ]
        )
 

        self.conv_24_25_dw = nn.Sequential(
            
            
                DepthSepConv2d(

                in_channels = 512,
                out_channels = (512, 1024),
                stride = (2, 1),
                padding = (1, 0),
                alpha = alpha
            
            ),
                
                SEBlock(
                    
                    channels = 1024,
                    reduct_ratio = reduct_ratio,
                    alpha = alpha 
                    
                )
                
        )

        self.conv_26_27_dw = nn.Sequential(
            
            
            DepthSepConv2d(
                
                in_channels = 1024,
                out_channels = (1024, 1024),
                stride = (1, 1), # seems to be error in paper -- they reference it as stride (2, 1), and not (1, 1) but that leads to the wrong output shape according to Table 1 -- ref: https://arxiv.org/pdf/1704.04861v1
                padding = (1, 0),
                alpha = alpha

            ),
            
            SEBlock(
                
                channels = 1024,
                reduct_ratio = reduct_ratio,
                alpha = alpha
                
            )
            
        )

        self.avgpool_24 = nn.AdaptiveAvgPool2d(output_size = (1, 1))
        self.fc = nn.Linear(in_features = int(1024 * alpha), out_features = 1000)


    def forward(self, x):

        x = self.transforms(x) # transformation via rho, resolution param
        x = self.conv_1_in(x)
        x = self.conv_2_3_dw(x)
        x = self.conv_4_5_dw(x)
        x = self.conv_6_7_dw(x) 
        x = self.conv_8_9_dw(x)
        x = self.conv_10_11_dw(x)
        x = self.conv_12_13_dw(x)
        x = self.conv_14_23_dw(x)
        x = self.conv_24_25_dw(x)
        x = self.conv_26_27_dw(x)
        x = torch.flatten(self.avgpool_24(x), start_dim = 1)
        x = self.fc(x)

        return x

class DepthSepConv2d(nn.Module):

    def __init__(self, in_channels:int, out_channels:tuple, stride:tuple, alpha = None, padding = (0, 0)):

        assert alpha is not None, ValueError(f'alpha must not be {type(None)}')

        super().__init__()
            
        groups = int(in_channels * alpha)
        in_channels = int(in_channels * alpha)
        out_channels = [int(x * alpha) for x in out_channels]

        self.conv_3x3 = BasicConv2d(

            in_channels = in_channels,
            out_channels = out_channels[0],
            groups = groups,
            stride = stride[0],
            padding = padding[0],

        ) 

        self.conv_1x1 = BasicConv2d(

            in_channels = out_channels[0],
            out_channels = out_channels[1],
            kernel_size = 1,
            stride = stride[1],
            padding = padding[1],

        )

    def forward(self, x):

        x = self.conv_3x3(x)
        x = self.conv_1x1(x)

        return x

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 0, groups = 1, alpha = None):

        super().__init__()

        if alpha is not None:

            out_channels = int(out_channels * alpha)

        self.conv = nn.Conv2d(
            
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            groups = groups

        )

        self.bn = nn.BatchNorm2d(num_features = out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x
