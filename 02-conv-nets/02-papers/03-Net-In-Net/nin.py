import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F

class NiN(nn.Module):
   
    '''
    
    Implemented, same as AlexNet, instead with 1x1 Convolutions! 
    
    '''
    
    def __init__(self):
        super().__init__()
     
        self.NIN1 = nn.Sequential(
        
            nn.Conv2d(kernel_size = 11, in_channels = 3, out_channels = 96, stride = 4),
            nn.ReLU(),
            nn.Conv2d(kernel_size = 1, in_channels = 96, out_channels = 96, stride = 1),
            nn.ReLU(),
            nn.Conv2d(kernel_size = 1, in_channels = 96, out_channels = 96, stride = 1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2) 
            
        ) 

        self.NIN2 = nn.Sequential(
            
            nn.Conv2d(kernel_size = 5, in_channels = 96, out_channels = 256, padding = 2),
            nn.ReLU(),
            nn.Conv2d(kernel_size = 1, in_channels = 256, out_channels = 256),
            nn.ReLU(),
            nn.Conv2d(kernel_size = 1, in_channels = 256, out_channels = 256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
            
            
        )
        
        self.NIN3 = nn.Sequential(
            
            nn.Conv2d(kernel_size = 3, in_channels = 256, out_channels = 384, padding = 1) ,
            nn.ReLU(),
            nn.Conv2d(kernel_size = 1, in_channels = 384, out_channels = 384),
            nn.ReLU(),
            nn.Conv2d(kernel_size = 1, in_channels = 384, out_channels = 384),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2) 
             
        ) 
     
        self.NIN4 = nn.Sequential(
            
            nn.Conv2d(kernel_size = 3, in_channels = 384, out_channels = 10, padding = 1),
            nn.ReLU(),
            nn.Conv2d(kernel_size = 1, in_channels = 10, out_channels = 10),
            nn.ReLU(),
            nn.Conv2d(kernel_size = 1, in_channels = 10, out_channels = 10),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size = 5),
            nn.Flatten()
        ) 
       
       
    def forward(self, x):
    
        x = self.NIN1(x)
        x = self.NIN2(x) 
        x = self.NIN3(x)
        x = self.NIN4(x)
        y = F.softmax(x)
             
        