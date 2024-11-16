import torch 
import torch.nn as nn
from mobileblock import MobileBlock
from ops import BasicConv2d

'''

Residual Connections are only applied to blocks that have same number of I/O channels, as should've been done in MobileNetV2.

The rest is implemneted as shown in Table 1 of -- https://arxiv.org/pdf/1905.02244


Some personal notes:

- Expansion Size defines the expressiveness of the model's learnt features, via the 3x3 depthwise convs after the 1x1 projection.
- We use an expansion size because we want to increase expressiveness of learnt features, while outside of that 3x3 depthwise conv, we remain at a lower dimensionality, here denoted by the count of out_channels.


TODO

- [ ] They didn't use SE blocks for every layer, check diagram and fix code!

Quesetions

- [ ] where the hell am i supposed to use the residual connectoins if we always do a reduction?

    Residual connections can be used when:

    Same spatial dimensions (same input height/width)
    Same number of output channels (#out)
    Stride (s) = 1

    Going through the table:

    112² × 16: First three layers no residual (first has stride=2, second changes channels from 16->24)
    56² × 24: Can use residual between the two bneck layers (same channels, stride=1)
    28² × 40: Can use residual between the three bneck layers (same channels, stride=1)
    14² × 80: Can use residual between all four bneck layers (same channels, stride=1)
    14² × 112: Can use residual between the two bneck layers (same channels, stride=1)
    7² × 160: Can use residual between all three bneck layers (same channels, stride=1)

'''

class MobileNetV3Large(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = BasicConv2d(

            in_channels = 3,
            out_channels = 16,
            kernel_size = 3,
            stride = 2,
            padding = 1

        )

        self.bneck2 = MobileBlock(
           
            in_channels = 16,
            out_channels = 16,
            kernel_size = 3,
            expansion_ch = 16,
            padding = 1,
            non_linearity = 'relu',
            residual = False,
            squeeze_excite = False

        )

        self.bneck3 = MobileBlock(

            in_channels = 16,
            out_channels = 24,
            kernel_size = 3,
            expansion_ch = 64,
            stride = 2,
            padding = 1,
            non_linearity = 'relu',
            residual = False,
            squeeze_excite = False

        )

        self.bneck4 = MobileBlock(

            in_channels = 24,
            out_channels = 24,
            kernel_size = 3,
            expansion_ch = 72,
            padding = 1,
            non_linearity = 'relu',
            residual = True,
            squeeze_excite = False

        )

        self.bneck5 = MobileBlock(

            in_channels = 24,
            out_channels = 40,
            kernel_size = 5,
            expansion_ch = 72,
            stride = 2,
            padding = 2,
            non_linearity = 'relu',
            residual = False

        )

        self.bneck6 = MobileBlock(

            in_channels = 40,
            out_channels = 40,
            kernel_size = 5,
            expansion_ch = 120,
            padding = 2,
            non_linearity = 'relu'

        )
        

        self.bneck7 = MobileBlock(

            in_channels = 40,
            out_channels = 40,
            kernel_size = 5,
            expansion_ch = 120,
            padding = 2,
            non_linearity = 'relu'

        )

        self.bneck8 = MobileBlock(

            in_channels = 40,
            out_channels = 80,
            kernel_size = 3,
            expansion_ch = 240,
            stride = 2,
            padding = 1,
            non_linearity = 'h_swish',
            residual = False,
            squeeze_excite = False
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.bneck2(x)
        x = self.bneck3(x)
        x = self.bneck4(x)
        x = self.bneck5(x)
        x = self.bneck6(x)
        x = self.bneck7(x)
        x = self.bneck8(x)


        return x
