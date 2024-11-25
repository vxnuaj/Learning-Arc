import torch
import torch.nn as nn

class SEBlock(nn.Module):

    '''
        
    c: count of input channels / output channels (same count)
    r: reduction ratio

    '''

    def __init__(self, channels, reduct_ratio):

        super().__init__()

        reduct_ch = channels // reduct_ratio

        self.avgpool = nn.AdaptiveAvgPool2d(

            output_size = (1, 1)
        
        )

        self.fc1 = nn.Linear(

            in_features = channels,
            out_features = reduct_ch


        )

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(

            in_features = reduct_ch,
            out_features = channels

        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x_scale = self.avgpool(x)

        x_scale = torch.flatten(
            
            x_scale,
            start_dim = 1

        )

        x_scale = self.fc1(x_scale)
        x_scale = self.relu(x_scale)
        x_scale = self.fc2(x_scale)
        x_scale = self.sigmoid(x_scale)
  
        x *= x_scale.view(x_scale.size(0), x_scale.size(1), 1, 1)
   
        return x
