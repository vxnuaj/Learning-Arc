import torch
import torch.nn as nn
from ops import EBlock, MBlock, SeperableConv2d, BasicConv2d

class Xception(nn.Module):

    def __init__(self, fc = False):

        self.fc = fc

        super().__init__()

        self.conv1 = BasicConv2d(

            in_channels = 3,
            out_channels = 32,
            kernel_size = 3,
            stride = 2,

        )

        self.conv2 = BasicConv2d(

            in_channels = 32,
            out_channels = 64,
            kernel_size = 3,

        )

        self.block_e3 = EBlock(

            in_channels = 64,
            out_channels = 128,
            init = True

        )

        self.block_e4 = EBlock(

            in_channels = 128,
            out_channels = 256,

        )

        self.block_e5 = EBlock(

            in_channels = 256,
            out_channels = 728

        )

        self.block_m6 = nn.Sequential(

            *[MBlock(

                in_channels = 728,
                out_channels = 728

                ) for _ in range(8)]

        )

        self.block_ex7 = EBlock(

            in_channels = 728,
            out_channels = [728, 1024]

        )

        self.conv8 = SeperableConv2d(

            in_channels = 1024,
            out_channels = 1536,
            act = True

        )

        self.conv9 = SeperableConv2d(

            in_channels = 1536,
            out_channels = 2048,
            act = True

        )

        self.avgpool10 = nn.AdaptiveAvgPool2d(

            output_size = (1, 1)

        )

        if self.fc:

            self.fc11 = nn.Linear(2048, 4096)
            self.fc12 = nn.Linear(4096, 4096)
            self.fc13 = nn.Linear(4096, 1000)

        else:

            self.fc11 = nn.Linear(2048, 1000)


    def forward(self, x):
    
        # init convs

        x = self.conv1(x)
        x = self.conv2(x)

        # entry flow | 3 blocks -- 6 convs

        x = self.block_e3(x)
        x = self.block_e4(x)
        x = self.block_e5(x)

        # middle flow | 8 blocks -- 24 convs

        x = self.block_m6(x)

        # exit flow | 1 block + 2 convs -- 4 convs
        
        x = self.block_ex7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.avgpool10(x)
        
        x = torch.flatten(
            
            x, 
            start_dim = 1

        )


        if self.fc:
            
            x = self.fc11(x)
            x = self.fc12(x)
            x = self.fc13(x)

        else:

            x = self.fc11(x)

        return x

