import torch
from torch import nn

'''

Implementation of DenseNet (BC), for 224 x 224 images.

Ref -- Table 1 @ https://arxiv.org/pdf/1608.06993

'''

class DenseNet(nn.Module):
    
    def __init__(self, k = 32):
       
        super().__init__()
       
        # note -- we introduce padding to ensure both, the conv and pooling cover all edges of the input feature map
        
        self.conv_1_in = BasicConv2d(3, out_channels = k * 2, kernel_size = 7, stride = 2, padding = 3, _in = True)  
        self.maxpool_1_in = nn.MaxPool2d(kernel_size = 3, stride = 2, padding =1)

    def forward(self, x):
        
        x = self.conv_1_in(x)
        x = self.maxpool_1_in(x)
        
        return x

class DenseBlock(nn.Module):
    
    def __init__(self, k:int, layers:int):
        
        super().__init__()
       
        self.layers = nn.ModuleList()
        
        in_channels = k   
        
        for _ in range(layers):
            
            self.layers.append(DenseConv2d(in_channels, k))
            in_channels += k
      
    def forward(self, x): 
        
        outputs = [x]
        
        for layer in self.layers:
            
            input = torch.cat(out, dim = 1)
            out = layer(input)
            outputs.append(out)
            
        return torch.cat(outputs, dim = 1)

class DenseConv2d(nn.Module):
   
    '''
    
    1x1 -> 3x3 -> out
    
    '''
    
    def __init__(self, in_channels, k):
        
        super().__init__()
       
        out_1 = 4 * k 
        
        self.conv_1x1 = BasicConv2d(in_channels, out_channels = out_1, kernel_size = 1)
        self.conv_3x3 = BasicConv2d(in_channels = out_1, out_channels = k, kernel_size = 3)
        
    def forward(self, x):
        
        x_out = self.conv_1x1(x)
        x_out = self.conv_3x3(x_out)
        
        return torch.cat(
           
            [x, x_out], 
            dim = 1
            
            )
        
class BasicConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 0, stride = 1, _in = False):
        
        super().__init__()
       
        self._in = _in 
        
        self.batchnorm = nn.BatchNorm2d(num_features = out_channels if self._in else in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(
            
            in_channels, 
            out_channels, 
            kernel_size = kernel_size, 
            stride = stride, 
            padding = padding
            
            )
        
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