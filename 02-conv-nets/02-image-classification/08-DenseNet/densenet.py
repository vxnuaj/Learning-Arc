import torch
from torch import nn

'''

Implementation of DenseNet-264 (BC), for 224 x 224 images.

Ref -- Table 1 @ https://arxiv.org/pdf/1608.06993

'''

class DenseNet(nn.Module):
   
   
    '''
    
    k: growth rate for each denseblock
    theta: degree of compression, where 0 < theta ≤ 1. 
    
    '''
    
    def __init__(self, k = 32, theta = .5):
       
        super().__init__()
       
        # note -- we introduce padding to ensure both, 
        # the conv and pooling cover all edges of the input feature map
        
        self.conv_1_in = BasicConv2d(
            
            in_channels = 3, 
            out_channels = k * 2, 
            kernel_size = 7, 
            stride = 2, 
            padding = 3, 
            _in = True
            
            )  

        self.maxpool_1_in = nn.MaxPool2d(
            
            kernel_size = 3, 
            stride = 2, 
            padding = 1
            
            )
       
        # DenseBlocks & Transition Layers
        
        self.dense_blk_1 = DenseBlock(
            
            k = k, 
            layers = 6, 
            in_channels = 2 * k,
            theta = theta 
            
            )
      
        self.tran_blk_1 = Transition(
           
            in_channels = 8 * k,
            theta = theta 
            
        ) 
       
        
        self.dense_blk_2 = DenseBlock(
           
            k = k, 
            layers = 12, 
            in_channels = (8 * k) * theta,
            theta = theta 
            
            )
     
        self.tran_blk_2 = Transition(
            
            in_channels = 12 * k + (8 * k) * theta,
            theta = theta
            
        ) 

        self.dense_blk_3 = DenseBlock(
            
            k = k,
            layers = 64, 
            in_channels = (12 * k + (8 * k) * theta) * theta, 
            theta = theta 
        )

        self.tran_blk_3 = Transition(
            
            in_channels = ((12 * k + (8 * k) * theta) * theta) + 64 * k,
            theta = theta 
            
        )
        
        self.dense_blk_4 = DenseBlock(
            
            k = k,
            layers = 48, 
            in_channels = (((12 * k + (8 * k) * theta) * theta) + 64 * k) * theta,
            theta = theta
            
        )
        
        self.avgpool_1_out = nn.AdaptiveAvgPool2d(output_size = (1, 1))

        self.fc = nn.Linear(
            
            in_features = int((((12 * k + (8 * k) * theta) * theta) + 64 * k) * theta),
            out_features = 1000
            
            )
        
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity = 'relu') # kaiming init.

    def forward(self, x):
        
        x = self.conv_1_in(x)
        x = self.maxpool_1_in(x)
        x = self.dense_blk_1(x)
        x = self.tran_blk_1(x)
        x = self.dense_blk_2(x) 
        x = self.tran_blk_2(x)
        x = self.dense_blk_3(x)
        x = self.tran_blk_3(x)
        x = self.avgpool_1_out(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.fc(x)

        return x

class DenseBlock(nn.Module):
    
    def __init__(self, k:int, layers:int, in_channels:int, theta: float):
        
        super().__init__()
       
        self.layers = nn.ModuleList()
        
        for _ in range(layers):
           
            self.layers.append(DenseConv2d(in_channels, k, theta))
            
            in_channels += k
      
    def forward(self, x):
       
        outputs = [x] 
        
        for _, layer in enumerate(self.layers):
          
            input = torch.cat(outputs, dim = 1)
            out = layer(input)
            outputs.append(out)
        
        return torch.cat(outputs, dim = 1)

class DenseConv2d(nn.Module):
   
    '''
    
    1x1 -> 3x3 -> out
    
    '''
   
    def __init__(self, in_channels, k, theta):
        
        super().__init__()
       
        out_1 = 4 * k * theta
   
        self.conv_1x1 = BasicConv2d(in_channels, out_channels = out_1, kernel_size = 1) 
        self.conv_3x3 = BasicConv2d(in_channels = out_1, out_channels = k, kernel_size = 3, padding = 1)
        
    def forward(self, x):
       
        x_out = self.conv_1x1(x)
        x_out = self.conv_3x3(x_out)
        
        return x_out

class Transition(nn.Module):
    
    def __init__(self, in_channels, theta:float):
       
        assert 0 < theta <= 1, ValueError('theta must be 0 < theta ≤ 1')
        
        super().__init__()
       
        out_channels = int(in_channels * theta)
     
        self.conv = BasicConv2d(
            
            in_channels = int(in_channels), 
            out_channels = int(out_channels),
            kernel_size = 1
            
            )
        
        self.avgpool = nn.AvgPool2d(
            
            kernel_size = 2,
            stride = 2
            
        )
         
       
    def forward(self, x):
        
        x = self.conv(x) 
        x = self.avgpool(x)
        
        return x 
         
class BasicConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 0, stride = 1, _in = False):
        
        super().__init__()
       
        self._in = _in 
       
         
        in_channels = int(in_channels)  
        out_channels = int(out_channels)
        
        self.batchnorm = nn.BatchNorm2d(
            
            num_features = out_channels 
            if self._in else 
            in_channels
            
            )
        
        self.relu = nn.ReLU()
        
        self.conv = nn.Conv2d(
            
            in_channels,
            out_channels, 
            kernel_size = kernel_size, 
            stride = stride, 
            padding = padding
            
            )
      
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity = 'relu' ) # initializing via kaiming init.
         
       
    def forward(self, x):
       
        if self._in:
            
            x = self.conv(x)
            x = self.batchnorm(x)
            x = self.relu(x) 
       
            return x 
        
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.conv(x)
        
        return x