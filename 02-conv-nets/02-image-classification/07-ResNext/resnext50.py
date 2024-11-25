import torch
import torch.nn as nn

class ResNext50(nn.Module):

    '''

    Implemented with Cardinality = 32 and Bottleneck width to be 4d, such that total input channel count 
    to a given ResNext block becomes 4 * 32 = 128.

    Input : N x 3 x 224 x 224, where N > 1, such that BatchNorm works properly.

    '''


    def __init__(self):

        super().__init__()
       
        self.conv1 = BasicConv2d(

            in_channels = 3, 
            out_channels = 64,
            kernel_size = 7,
            stride = 2,
            padding = 3

        )  

        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.conv2_x = nn.Sequential(

            NextBlock( 

                in_channels = 64,
                out_1 = 64,
                out_2 = 64,
                out_3 = 256,
                C = 32,
                project = True

            ),    

            NextBlock(

                in_channels = 256,
                out_1 = 64,
                out_2 = 64,
                out_3 = 256,
                C = 32,
            ),

            NextBlock(

                in_channels = 256,
                out_1 = 64,
                out_2 = 64,
                out_3 = 256,
                C = 32

            )

        )

        self.conv3_x = nn.Sequential(

            NextBlock(

                in_channels = 256,
                out_1 = 128,
                out_2 = 128,
                out_3 = 512,
                C = 32,
                blck_in = True,
                project = True

            ),

            NextBlock(

                in_channels = 512,
                out_1 = 128,
                out_2 = 128,
                out_3 = 512,
                C = 32

            ),

            NextBlock(

                in_channels = 512,
                out_1 = 128,
                out_2 = 128,
                out_3 = 512,
                C = 32

            ),

            NextBlock(

                in_channels = 512,
                out_1 = 128,
                out_2 = 128,
                out_3 = 512,
                C = 32

            )

        )


        self.conv4_x = nn.Sequential(

            NextBlock(

                in_channels = 512,
                out_1 = 256,
                out_2 = 256,
                out_3 = 1024,
                C = 32,
                blck_in = True,
                project = True

            ),


            NextBlock(

                in_channels = 1024,
                out_1 = 256,
                out_2 = 256,
                out_3 = 1024,
                C = 32,
            ),

            NextBlock(

                in_channels = 1024,
                out_1 = 256,
                out_2 = 256,
                out_3 = 1024,
                C = 32,
            ),

            NextBlock(

                in_channels = 1024,
                out_1 = 256,
                out_2 = 256,
                out_3 = 1024,
                C = 32,
            ),

            NextBlock(

                in_channels = 1024,
                out_1 = 256,
                out_2 = 256,
                out_3 = 1024,
                C = 32,
            ),

            NextBlock(

                in_channels = 1024,
                out_1 = 256,
                out_2 = 256,
                out_3 = 1024,
                C = 32,
            ),

        )


        self.conv5_x = nn.Sequential(


            NextBlock(

                in_channels =  1024,
                out_1 = 512,
                out_2 = 512,
                out_3 = 2048,
                C = 32,
                blck_in = True,
                project = True

            ),

            NextBlock(

                in_channels = 2048,
                out_1 = 512,
                out_2 = 512,
                out_3 = 2048,
                C = 32

            ),

            NextBlock(

                in_channels = 2048,
                out_1 = 512,
                out_2 = 512,
                out_3 = 2048,
                C = 32

            )

        )

        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features = 2048, out_features = 1000)

    def forward(self, x):

        x = self.conv1(x) # N, 64, 112, 112
        x = self.maxpool(x) # N, 64, 56, 56
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.fc(x)

        return x


class NextBlock(nn.Module):

    '''

    C: Cardinality for the ResNext Block
    out_1: Output for the first layer of all branches (1 x 1 bottlenecks, denoted as "xd" in the paper, where x is the # of input channels to each branch
    out_2: Output for the second layer of all branches (3 x 3)
    out_3: Output for the third layer of all branches (1 x 1), applied post-concatenation of all C branches.

    '''

    def __init__(self, in_channels, out_1, out_2, out_3, C, blck_in = False, project = False):

        super().__init__()

        self.project = project

        if blck_in:
            stride = 2
        else:
            stride = 1


        self.conv_1x1 = BasicConv2d(

            in_channels = in_channels,
            out_channels = out_1,
            kernel_size = 1,
            
        )

        self.conv_3x3 = BasicConv2d(

            in_channels = out_1,
            out_channels = out_2,
            kernel_size = 3,
            stride = stride,
            padding = 1,
            groups = C

        )

        self.conv_1x1_2 = BasicConv2d(

            in_channels = out_2,
            out_channels = out_3,
            kernel_size = 1,
            blck_out = True

        )


        self.dim_project = BasicConv2d(

            in_channels = in_channels,
            out_channels = out_3,
            kernel_size = 1,
            stride = stride

        )

        self.identity = nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):

        if self.project:
            x_res = self.dim_project(x)

        else:
            x_res = self.identity(x)

        x = self.conv_1x1(x)
        x = self.conv_3x3(x)
        x = self.conv_1x1_2(x)

        return self.relu(x_res + x)
                


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 0, groups = 1, blck_out = False):
       
        super().__init__()

        self.blck_out = blck_out

        self.conv = nn.Conv2d(
        
            in_channels = in_channels, 
            out_channels = out_channels, 
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            groups = groups

        )

        self.batchnorm = nn.BatchNorm2d(num_features = out_channels)

        self.relu = nn.ReLU()


    def forward(self, x):

        x = self.conv(x)
        x = self.batchnorm(x)

        if self.blck_out:

            return x

        x = self.relu(x)

        return x
es