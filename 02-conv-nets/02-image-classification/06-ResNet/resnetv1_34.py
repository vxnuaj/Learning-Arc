import torch
import torch.nn as nn

class ResNetV1(nn.Module):
   
    '''
    
    Input: Nx3x224x224 where N > 1, such that BatchNorm actually works lol.
    
    '''
    
    def __init__(self):
        
        super().__init__()
        
        self.conv = BasicConv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
      
        self.resblock_64 = nn.Sequential(
            
            ResBlock(in_channels = 64, out_channels = 64),
            ResBlock(in_channels = 64, out_channels = 64),
            ResBlock(in_channels = 64, out_channels = 64), 
            
        )
        
        self.resblock_128 = nn.Sequential(
            
            ResBlock(in_channels = 64, out_channels = 128, dim_reduct = True),
            ResBlock(in_channels = 128, out_channels = 128),
            ResBlock(in_channels = 128, out_channels = 128),
            ResBlock(in_channels = 128, out_channels = 128) 
            
        )
       
        self.resblock_256 = nn.Sequential(
            
            ResBlock(in_channels = 128, out_channels = 256, dim_reduct = True),
            ResBlock(in_channels = 256, out_channels = 256),
            ResBlock(in_channels = 256, out_channels = 256),
            ResBlock(in_channels = 256, out_channels = 256),
            ResBlock(in_channels = 256, out_channels = 256),
            ResBlock(in_channels = 256, out_channels = 256)

        ) 
        
        
        self.resblock_512 = nn.Sequential(
            
            ResBlock(in_channels = 256, out_channels = 512, dim_reduct = True),
            ResBlock(in_channels = 512, out_channels = 512),
            ResBlock(in_channels = 512, out_channels = 512) 
            
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
    
    def __init__(self, in_channels, out_channels, dim_reduct:bool = False):
        
        super().__init__() 
      
        self.dim_reduct = dim_reduct 
       
        self.resconnect = nn.Identity()
        self.relu = nn.ReLU()

        if dim_reduct:
           
            self.resblck_reduct = nn.Sequential(
                
                BasicConv2d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1),
                BasicConv2d(out_channels, out_channels, kernel_size = 3, padding = 'same', act = False)
                 
            ) 
            
            self.resconnect_reduct = BasicConv2d(in_channels, out_channels, kernel_size = 1, stride = 2)
         
        else:  
            
            self.resblock = nn.Sequential(
                
                BasicConv2d(in_channels, out_channels, kernel_size = 3, padding = 'same'),
                BasicConv2d(out_channels, out_channels, kernel_size = 3, padding = 'same', act = False)
                
            )
            
                        
    def forward(self, x):
        
        if self.dim_reduct:
            
            x_out = self.relu(self.resblck_reduct(x) + self.resconnect_reduct(x))
       
        else:
            
            x_out = self.relu(self.resblock(x) + self.resconnect(x))
            
        return x_out

        
        
class BasicConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, act:bool = True):
        
        super().__init__()
       
        self.act = act 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, padding = padding)
        self.bnorm = nn.BatchNorm2d(num_features = out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        
        x = self.conv(x)
        x = self.bnorm(x)
        
        if self.act is False:
            return x
        
        x = self.relu(x)
        
        return x
