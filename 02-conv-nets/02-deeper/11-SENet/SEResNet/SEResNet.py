import torch
import torch.nn as nn
from SEBlock import SEBlock

'''

Implementation of SEResNet, ResNetV2 with Squeeze-and-Excitation Blocks

'''


class SEResNet(nn.Module):
   
    '''
    
    Input: Nx3x224x224 where N > 1, such that BatchNorm actually works lol.
    reduct_ratio: Reduction Ratio for the SEBlock
    
    '''
    
    def __init__(self, reduct_ratio):
        
        super().__init__()
        
        self.conv = BasicConv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3, res = False)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
      
        self.resblock_64 = nn.Sequential(
            
            ResBlock(in_channels = 64, out_channels = 64, reduct_ratio = reduct_ratio),
            ResBlock(in_channels = 64, out_channels = 64, reduct_ratio = reduct_ratio),
            ResBlock(in_channels = 64, out_channels = 64, reduct_ratio = reduct_ratio), 
            
        )
        
        self.resblock_128 = nn.Sequential(
            
            ResBlock(in_channels = 64, out_channels = 128, reduct_ratio = reduct_ratio, dim_reduct = True), 
            ResBlock(in_channels = 128, out_channels = 128, reduct_ratio = reduct_ratio),
            ResBlock(in_channels = 128, out_channels = 128, reduct_ratio = reduct_ratio),
            ResBlock(in_channels = 128, out_channels = 128, reduct_ratio = reduct_ratio) 
            
        )
       
        self.resblock_256 = nn.Sequential(
            
            ResBlock(in_channels = 128, out_channels = 256, reduct_ratio = reduct_ratio, dim_reduct = True),
            ResBlock(in_channels = 256, out_channels = 256, reduct_ratio = reduct_ratio),
            ResBlock(in_channels = 256, out_channels = 256, reduct_ratio = reduct_ratio),
            ResBlock(in_channels = 256, out_channels = 256, reduct_ratio = reduct_ratio),
            ResBlock(in_channels = 256, out_channels = 256, reduct_ratio = reduct_ratio),
            ResBlock(in_channels = 256, out_channels = 256, reduct_ratio = reduct_ratio)

        ) 

        
        self.resblock_512 = nn.Sequential(
            
            ResBlock(in_channels = 256, out_channels = 512, reduct_ratio = reduct_ratio, dim_reduct = True),
            ResBlock(in_channels = 512, out_channels = 512, reduct_ratio = reduct_ratio),
            ResBlock(in_channels = 512, out_channels = 512, reduct_ratio = reduct_ratio) 
            
        )
       
        self.avgpool = nn.AdaptiveAvgPool2d(output_size = (1,1))
        self.fc = nn.Linear(in_features = 512, out_features = 1000)
        
   
    def forward(self, x):
       
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.resblock_64(x)
        x = self.resblock_128(x)
        x = self.resblock_256(x)
        x = self.resblock_512(x)
        x = torch.flatten(self.avgpool(x), start_dim = 1)
        x = self.fc(x)
        return x
     
       
class ResBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, reduct_ratio, dim_reduct:bool = False): 
        
        super().__init__() 
      
        self.dim_reduct = dim_reduct 
      
        self.seblock = SEBlock(

            channels = out_channels,
            reduct_ratio = reduct_ratio

        )

        self.resconnect = nn.Identity()
        self.relu = nn.ReLU()

        if dim_reduct:
           
            self.resblck_reduct = nn.Sequential(
                
                BasicConv2d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1),
                BasicConv2d(out_channels, out_channels, kernel_size = 3, padding = 'same')
                 
            ) 
            
            self.resconnect_reduct = BasicConv2d(in_channels, out_channels, kernel_size = 1, stride = 2)
         
        else:  
            
            self.resblock = nn.Sequential(
                
                BasicConv2d(in_channels, out_channels, kernel_size = 3, padding = 'same'),
                BasicConv2d(out_channels, out_channels, kernel_size = 3, padding = 'same')
                
            )
            
                        
    def forward(self, x):
        
        if self.dim_reduct:
         
              
            x_out = self.resblck_reduct(x)
            x_out = self.seblock(x_out)
            x = self.resconnect_reduct(x)
          
            x_out += x
       
        else:
 

            x_out = self.resblock(x)
            x_out = self.seblock(x_out)
            x = self.resconnect(x)

            x_out += x

        return x_out

class BasicConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, res = True):
        
        super().__init__()
       
        self.res = res 
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, padding = padding)
        self.relu = nn.ReLU()
       
        if not self.res:
            self.bnorm = nn.BatchNorm2d(num_features = out_channels)
        else:
            self.bnorm = nn.BatchNorm2d(num_features = in_channels)
            
            
    def forward(self, x):
        
        if not self.res: 
            x = self.conv(x)
            x = self.bnorm(x)
            x = self.relu(x)
        
        else:
            x = self.bnorm(x)
            x = self.relu(x)
            x = self.conv(x)
        
        return x
