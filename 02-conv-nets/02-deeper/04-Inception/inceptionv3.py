import torch
import torch.nn as nn

class InceptionV3(nn.Module):
    '''
       
    Inputs: Bx3x299x299
        
    ''' 
    
    def __init__(self):
        super().__init__()
        
       
        self.head = nn.Sequential(
            
            Conv2dBN(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 2), # -> 32 x 149 x 149
            Conv2dBN(in_channels = 32, out_channels = 32, kernel_size = 3), # 32 x 147 x 147
            Conv2dBN(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1), # 64 x 147 x147
            nn.MaxPool2d(kernel_size = 3, stride = 2), # 64 x 73 x 73
            Conv2dBN(in_channels = 64, out_channels = 80, kernel_size = 1),  # 80 x 73 x 73
            Conv2dBN(in_channels = 80, out_channels = 192, kernel_size = 3), # 192 x 71 x 71
            nn.MaxPool2d(kernel_size = 3, stride = 2) # 192 x 35 x 35
             
        )
     
        # inception A
        self.body1 = nn.Sequential(
            InceptionBlockA(in_channels = 192, out_1 = 64, out_2 = (48, 64), out_3 = (64, 96, 96), out_4 = 32), 
            InceptionBlockA(in_channels = 256, out_1 = 64, out_2 = (48, 64), out_3 = (64, 96, 96), out_4 = 64), 
            InceptionBlockA(in_channels = 288, out_1 = 64, out_2 = (48, 64), out_3 = (64, 96, 96), out_4 = 64)
        ) 
        
        self.body2 = InceptionBlockB(in_channels = 288, out_1 = 384, out_2 = (64, 64, 96)) # grid size reduction
      
        self.body3 = nn.Sequential(
            
            InceptionBlockC(in_channels = 768, out_1 = 192, out_2 = (128, 128, 192), out_3 = (128, 128, 128, 128, 192), out_4 = 192),
            InceptionBlockC(in_channels = 768, out_1 = 192, out_2 = (160, 160, 192), out_3 = (160, 160, 160, 160, 192), out_4 = 192),
            InceptionBlockC(in_channels = 768, out_1 = 192, out_2 = (160, 160, 192), out_3 = (160, 160, 160, 160, 192), out_4 = 192),
            InceptionBlockC(in_channels = 768, out_1 = 192, out_2 = (192, 192, 192), out_3 = (192, 192, 192, 192, 192), out_4 = 192)
        )
        
        self.aux = InceptionAux(in_channels = 768)
        self.body4 = InceptionBlockD(in_channels = 768, out_1 = (192, 320), out_2 = (192, 192, 192, 192)) # grid size reduction -> 1, 1280, 8, 8
      
        self.body5 = nn.Sequential(
            
            InceptionBlockE(in_channels = 1280, out_1 = 320, out_2 = (384, 384, 384), out_3 = (448, 384, 384, 384), out_4 = (192)),
            InceptionBlockE(in_channels = 2048, out_1 = 320, out_2 = (384, 384, 384), out_3 = (448, 384, 384, 384), out_4 = (192))
        
        )

        self.stem = nn.Sequential(
            
            nn.AdaptiveAvgPool2d(output_size = (1, 1)),
            nn.Dropout( p = .5 ),
            nn.Flatten(),
            nn.Linear(in_features = 2048, out_features = 1000)
            
        )
        
         
    def forward(self, x):
        x = self.head(x) # -> 192, 35, 35
        x = self.body1(x) # -> 288, 35, 35
        x = self.body2(x) # -> 768 x 17 x 17
        x = self.body3(x) # -> 768 x 17 x 17
        aux_x = self.aux(x)  
        x = self.body4(x) # -> 1280 x 8 x 8
        x = self.body5(x) # -> 1280 x 8 x 8
        x = self.stem(x) # -> 1000
        
        print(x.size())
        
        return x
        
        
class Conv2dBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0):
        
        super().__init__()
       
        self.conv =  nn.Conv2d(
                   
                    in_channels = in_channels,
                    out_channels = out_channels,
                    kernel_size = kernel_size,
                    stride = stride,
                    padding=padding
                    
                    )

        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
       
    def forward(self, x):
        
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        
        return x

class InceptionBlockA(nn.Module):
        
    def __init__(
        self,
        in_channels: int,
        out_1: int,
        out_2: tuple,
        out_3: tuple,
        out_4: int,
    ):
      
        self._val_out(
            out_1, 
            out_2, 
            out_3,
            out_4,
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
            
            Conv2dBN(
                in_channels, 
                out_channels = out_1, 
                kernel_size = 1
                ),
            
        )
       
        self.b2 = nn.Sequential(
            
            Conv2dBN(
                in_channels, 
                out_channels = out_2[0], 
                kernel_size = 1
                ),
           
            Conv2dBN(
                in_channels = out_2[0], 
                out_channels = out_2[1], 
                kernel_size = 5, 
                padding = 'same'
                ),
            
        ) 

        self.b3 = nn.Sequential(
            
            Conv2dBN(
                in_channels, 
                out_channels = out_3[0], 
                kernel_size = 1),

            Conv2dBN(
                
                in_channels = out_3[0],
                out_channels = out_3[1],
                kernel_size = 3,
                padding = 'same' 
                
            ),

            Conv2dBN(
                in_channels = out_3[1], 
                out_channels = out_3[2], 
                kernel_size = 3, 
                padding = 'same'
                ),
            
        )
 
        
        self.b4 = nn.Sequential(
            
            nn.AvgPool2d(
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                ),

            Conv2dBN(
                in_channels, 
                out_channels = out_4,
                kernel_size = 1,
                ),
            
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


class InceptionBlockB(nn.Module):
    
    def __init__(self, in_channels, out_1, out_2):
        
        super().__init__()
        
        self.b1 = nn.Sequential(
           
            # Conv2dBN(in_channels, out_channels = out_1, kernel_size = 1) <- was mentioned in paper, but not in official implementation ... see figure 10 and the keras / pytorch implementation for comparison
            Conv2dBN(in_channels, out_channels = out_1, kernel_size = 3, stride = 2)
            
        )
       
        self.b2 = nn.Sequential(
           
            Conv2dBN(in_channels, out_channels = out_2[0], kernel_size = 1),
            Conv2dBN(in_channels = out_2[0], out_channels = out_2[1], kernel_size = 3, padding = 1),
            Conv2dBN(in_channels = out_2[1], out_channels = out_2[2], kernel_size = 3, stride = 2) 
            
        ) 
       
        self.b3 = nn.MaxPool2d(kernel_size = 3, stride = 2) 
        
    def forward(self, x):
        
        x1 = self.b1(x)
        x2 = self.b2(x)
        x3 = self.b3(x)
        
        return torch.cat(
            [x1, x2, x3],
            dim = 1
        )
        
class InceptionBlockC(nn.Module):
    
    def __init__(self, in_channels, out_1, out_2, out_3, out_4):
        
        super().__init__()
       
        self.b1 = Conv2dBN(in_channels, out_channels = out_1, kernel_size = 1)
        self.b2 = nn.Sequential(
            
            Conv2dBN(in_channels, out_channels = out_2[0], kernel_size = 1),
            Conv2dBN(in_channels = out_2[0], out_channels = out_2[1], kernel_size = (1, 7), padding = (0, 3)), # TODO why are we using this padding setting??
            Conv2dBN(in_channels = out_2[1], out_channels = out_2[2], kernel_size = (7, 1), padding = (3, 0))
        ) 
        
        self.b3 = nn.Sequential(
            
            Conv2dBN(in_channels, out_channels = out_3[0], kernel_size = 1),
            Conv2dBN(in_channels = out_3[0], out_channels = out_3[1], kernel_size = (7, 1), padding = (3, 0)),
            Conv2dBN(in_channels = out_3[1], out_channels = out_3[2], kernel_size = (1, 7), padding = (0, 3)),
            Conv2dBN(in_channels = out_3[2], out_channels = out_3[3], kernel_size = (7, 1), padding = (3, 0)),
            Conv2dBN(in_channels = out_3[3], out_channels = out_3[4], kernel_size = (1, 7), padding = (0, 3))

        )
        
        self.b4 = nn.Sequential(
            
            nn.AvgPool2d(kernel_size = 3, stride = 1, padding = 1),
            Conv2dBN(in_channels, out_channels = out_4, kernel_size = 1) 
            
        )
        
    def forward(self, x):
       
        x1 = self.b1(x) 
        x2 = self.b2(x)
        x3 = self.b3(x)
        x4 = self.b4(x)
        
        return torch.cat(
            [x1, x2, x3, x4],
            dim = 1
        )
    
class InceptionBlockD(nn.Module):
    def __init__(self, in_channels, out_1, out_2):
        super().__init__()
        
        self.b1 = nn.Sequential(
           
            Conv2dBN(in_channels, out_channels = out_1[0], kernel_size = 1),
            Conv2dBN(in_channels = out_1[0], out_channels = out_1[1], kernel_size = 3, stride = 2)
            
        ) 
        
        self.b2 = nn.Sequential(
            
            Conv2dBN(in_channels, out_channels = out_2[0], kernel_size = 1),
            Conv2dBN(in_channels = out_2[0], out_channels = out_2[1], kernel_size = (1, 7), padding = (0, 3)) ,
            Conv2dBN(in_channels =  out_2[1], out_channels =  out_2[2] ,kernel_size = (7, 1), padding = (3, 0)),
            Conv2dBN(in_channels =  out_2[2], out_channels = out_2[3], kernel_size = 3, stride = 2)
            
        )
        
        self.b3 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        
    def forward(self, x):
        
        x1 = self.b1(x) 
        x2 = self.b2(x)
        x3 = self.b3(x)
        
        return torch.cat(
            [x1, x2, x3],
            dim = 1
            )
            
        
class InceptionAux(nn.Module):
    
    def __init__(self, in_channels):
      
        super().__init__()
        
        self.avgpool = nn.AvgPool2d(kernel_size = 5, stride = 3)
        self.conv1 = Conv2dBN(in_channels = in_channels, out_channels = 128, kernel_size = 1)
        self.conv2 = Conv2dBN(in_channels = 128, out_channels = 768, kernel_size = 5)
        self.fc = nn.Linear(in_features = 768, out_features = 1000)
    
    def forward(self, x):
        
        x = self.avgpool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim = 1) 
        x = self.fc(x)              
               
        return x
  
   
class InceptionBlockE(nn.Module):
    
    def __init__(self, in_channels, out_1:int, out_2:tuple, out_3:tuple, out_4:int):
        super().__init__()
        
        self.b1 = Conv2dBN(in_channels, out_channels = out_1, kernel_size = 1)
        
        self.b2_1 = Conv2dBN(in_channels, out_channels = out_2[0], kernel_size = 1)
        self.b2_2a = Conv2dBN(in_channels = out_2[0], out_channels = out_2[1], kernel_size = (1, 3), padding = (0, 1))
        self.b2_2b = Conv2dBN(in_channels = out_2[1], out_channels = out_2[2], kernel_size = (3, 1), padding = (1, 0)) 
        
        self.b3 = nn.Sequential(
            
            Conv2dBN(in_channels, out_channels = out_3[0], kernel_size = 1),
            Conv2dBN(in_channels = out_3[0], out_channels = out_3[1], kernel_size = 3, padding =1),
            
        )
       
        self.b3_2a = Conv2dBN(in_channels = out_3[1], out_channels = out_3[2], kernel_size = (1, 3), padding = (0, 1))
        self.b3_2b = Conv2dBN(in_channels = out_3[2], out_channels = out_3[3], kernel_size = (3, 1), padding = (1, 0))
            
        self.b4 = nn.Sequential(
            
            Conv2dBN(in_channels, out_channels = out_4, kernel_size = 1),
            nn.AvgPool2d(kernel_size = (3), stride = 1, padding = 1) 
             
        ) 
        
    def forward(self, x):
        
        x1 = self.b1(x)
        
        x2 = self.b2_1(x)
        x2 = torch.cat(
            [self.b2_2a(x2), 
            self.b2_2b(x2)],
            dim = 1
        )
                 
        x3 = self.b3(x)
        x3 = torch.cat(
            [self.b3_2a(x3),
             self.b3_2b(x3)],
            dim = 1
        )
        
        x4 = self.b4(x)
        
        return torch.cat(
            [x1, x2, x3, x4],
            dim = 1
            )
        