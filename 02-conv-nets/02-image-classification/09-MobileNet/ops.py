import torch.nn as nn

class BasicConv2d(nn.Module):

    def __init__(

        self, 
        in_channels,
        out_channels,
        kernel_size,
        stride = 1,
        padding = 0,
        groups = 1,
        batch_norm = True,
        non_linearity = None

    ):

        super().__init__()

        if non_linearity:
   
            self.non_linearity = non_linearity.lower()

        else:

            self.non_linearity = None

        self.conv = nn.Conv2d(

            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            groups = groups

            )

        if batch_norm:

            self.batch_norm = nn.BatchNorm2d(

                num_features = out_channels 

            )

        else:

            self.batch_norm = None

        if self.non_linearity == 'h_swish':

            self.non_linearity = nn.Hardswish()

        elif self.non_linearity == 'relu':

            self.non_linearity = nn.ReLU()

        else:

            self.non_linearity = None

    def forward(self, x):

        x = self.conv(x)
        
        if self.batch_norm:
            
            x = self.batch_norm(x)

        if self.non_linearity:

            x = self.non_linearity(x)

        return x
