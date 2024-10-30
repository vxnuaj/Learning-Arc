import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock(nn.Module):
   
    '''
    
    outX: output channels for the given Xth branch. If outX is hint == tuple: it must be defined for 2 layers elif: 1 layer 
    
    '''
    
    def __init__(
        self,
        in_channels: int,
        out_1: int,
        out_2: tuple,
        out_3: tuple,
        out_4: int
    ):
        
        self._val_out(
            out_1, 
            out_2, 
            out_3,
            out_4
        ) 
        
        super().__init__()
       
        '''
        b1 is the leftmost branch
        b2 is the second leftmost branch
        ... 
        ...
        etc 
        ''' 
       
        self.b1 = nn.Sequential(
            
            nn.Conv2d(
                in_channels, 
                out_channels = out_1, 
                kernel_size = 1
                ),
            
            nn.ReLU()
        )
        
        self.b2 = nn.Sequential(
            
            nn.Conv2d(
                in_channels, 
                out_channels = out_2[0], 
                kernel_size = 1),

            nn.ReLU(),

            nn.Conv2d(
                in_channels = out_2[0], 
                out_channels = out_2[1], 
                kernel_size = 3, 
                padding = 'same'
                ),
            
            nn.ReLU()
        )
        
        self.b3 = nn.Sequential(
            
            nn.Conv2d(
                in_channels, 
                out_channels = out_3[0], 
                kernel_size = 1
                ),
           
            nn.ReLU(), 
            
            nn.Conv2d(
                in_channels = out_3[0], 
                out_channels = out_3[1], 
                kernel_size = 5, 
                padding = 'same'
                ),
            
            nn.ReLU() 
            
        ) 
        
        self.b4 = nn.Sequential(
            
            nn.MaxPool2d(
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                ),

            nn.Conv2d(
                in_channels, 
                out_channels = out_4,
                kernel_size = 1,
                ),
            
            nn.ReLU() 
            
        )
    
    def forward(self, x):
        x1 = self.b1(x)
        x2 = self.b2(x)
        x3 = self.b3(x)
        x4 = self.b4(x)
      
        return torch.cat(
            [x1, x2, x3, x4],
            dim=1
        )

    def _val_out(self, out1, out2, out3, out4):
        assert isinstance(out1, int), ValueError('out1 must be type int')
        assert isinstance(out2, tuple) and all(isinstance(i, int) for i in out2), ValueError('out2 must be a tuple of ints')
        assert isinstance(out3, tuple) and all(isinstance(i, int) for i in out3), ValueError('out3 must be a tuple of ints')
        assert isinstance(out4, int), ValueError('out4 must be type int') 

class InceptionAux(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        '''
        avgpool:
            input: 14 x 14 
            output: 4 x 4 
        ''' 
      
        self.aux = nn.Sequential(
            
                nn.AvgPool2d(kernel_size = 5, stride = 3), # -> 4x4
                nn.Conv2d(in_channels, out_channels=128, kernel_size=1), # -> 128 x 4 x 4
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 1024), # B x 2048
                nn.ReLU(),
                nn.Dropout(p=0.7), 
                nn.Linear(1024, 1000) # B x 1000
       
        )  
        
    def forward(self, x):
        
        x = self.aux(x) 
        
        return x

class InceptionV1(nn.Module):
    def __init__(self):
        
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.body1 = nn.Sequential(
            InceptionBlock(192, 64, (96, 128), (16, 32), 32),
            InceptionBlock(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            InceptionBlock(480, 192, (96, 208), (16, 48), 64)
            
        )
        
        self.aux1 = InceptionAux(512)
        
        self.body2 = nn.Sequential(
            
            InceptionBlock(512, 160, (112, 224), (24, 64), 64),
            InceptionBlock(512, 128, (128, 256), (24, 64), 64),
            InceptionBlock(512, 112, (114, 288), (32, 64), 64),
        )
        
        self.aux2 = InceptionAux(528)
        
        self.body3 = nn.Sequential(  
            InceptionBlock(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            InceptionBlock(832, 256, (160, 320), (32, 128), 128),
            InceptionBlock(832, 384, (192, 384), (48, 128), 128)
        )

        self.head = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=1),
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=1024, out_features=1000)
        )
        
    def forward(self, x):

        x = self.stem(x)
        x = self.body1(x)
        aux1_x = self.aux1(x)
        x = self.body2(x)
        aux2_x = self.aux2(x)
        x = self.body3(x)
        x = self.head(x)

        return x, aux1_x, aux2_x