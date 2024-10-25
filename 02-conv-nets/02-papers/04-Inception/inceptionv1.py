import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO
# - define main inceptionv1 class
# - [ ] Learn how the auxiliary classifier works and how it avoids vanishing gradients
#

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
                ) 
        )
        
        self.b2 = nn.Sequential(
            
            nn.Conv2d(
                in_channels, 
                out_channels = out_2[0], 
                kernel_size = 1),

            nn.Conv2d(
                in_channels = out_2[0], 
                out_channels = out_2[1], 
                kernel_size = 3, 
                padding = 'same'
                ) 
        )
        
        self.b3 = nn.Sequential(
            
            nn.Conv2d(
                in_channels, 
                out_channels = out_3[0], 
                kernel_size = 1
                ),
            
            nn.Conv2d(
                in_channels = out_3[0], 
                out_channels = out_3[1], 
                kernel_size = 5, 
                padding = 'same'
                )
            
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
                )
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
        self.avgpool = nn.AvgPool2d(kernel_size = 5, stride = 3)
        self.conv = nn.Conv2d(in_channels, out_channels=128, kernel_size=1) 
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.dropout = nn.Dropout(p=0.7)
        self.fc2 = nn.Linear(1024, 1000)
        
    def forward(self, x):
        x = self.avgpool(x)  # -> 4x4
        x = self.conv(x)  # -> 128 x 4 x 4
        x = torch.flatten(x, start_dim=1)  # -> B X 2048
        x = self.fc1(x)  # -> B X 1024
        x = self.dropout(x)
        x = self.fc2(x)  # -> B X 1000
        # softmax activation (not in here as nn.CrossEntropyLoss automatically computes Softmax Activation)
        return x

class InceptionV1(nn.Module):
    def __init__(self):
        
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            padding = 3,
            stride=2
        )
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding = 1
        )
        self.localresnorm1 = nn.LocalResponseNorm(
            size=5,
            alpha=0.0001,
            beta=0.75,
            k=1
        )  # set to default params, as not defined within the paper
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=192,
            kernel_size=3,
            padding = 1
        )
        self.localresnorm2 = nn.LocalResponseNorm(
            size=5,
            alpha=0.0001,
            beta=0.75,
            k=1
        )  # set to default params, as not defined within the paper
        self.maxpool2 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding = 1
        )
        
        self.inception3a = InceptionBlock(
            in_channels=192, 
            out_1=64,
            out_2=(96, 128),
            out_3=(16, 32),
            out_4=32
        )  # -> 256 Channels
       
        self.inception3b = InceptionBlock(
            in_channels = 256,
            out_1 = 128,
            out_2 = (128, 192),
            out_3 = (32, 96),
            out_4 = 64
        ) # -> 480 Channels
       
        self.maxpool3 = nn.MaxPool2d(
            kernel_size = 3,
            stride = 2,
            padding = 1
        ) 
        
        self.inception4a = InceptionBlock(
            in_channels = 480, 
            out_1 =  192,
            out_2 = (96, 208),
            out_3 = (16, 48),
            out_4 = 64
        ) # -> 512 Channels
       
        self.aux1 = InceptionAux(
            in_channels = 512
        ) # auxiliary classifier to accumulate gradients midway through network, avoiding vanishing gradient problem.
        
        self.inception4b = InceptionBlock(
            in_channels = 512,
            out_1 = 160,
            out_2 = (112, 224),
            out_3 = (24, 64),
            out_4 = 64
        ) # -> 512 Channels
       
        self.inception4c = InceptionBlock(
            in_channels = 512,
            out_1 = 128,
            out_2 = (128, 256),
            out_3 = (24, 64),
            out_4 = 64 
        ) # -> 512 Channels
        
        self.inception4d = InceptionBlock(
            in_channels =  512,
            out_1 = 112,
            out_2 = (114, 288),
            out_3 = (32, 64),
            out_4 = 64 
        ) # -> 528 Channels
      
        self.aux2 = InceptionAux(
            in_channels = 528
        ) # auxiliary classifier to accumulate gradients midway through network, avoiding vanishing gradient problem.
     
        self.inception4e = InceptionBlock(
            in_channels = 528 ,
            out_1 = 256,
            out_2 = (160, 320),
            out_3 = (32, 128),
            out_4 = 128
        ) # -> 832 Channels
       
        self.maxpool5 = nn.MaxPool2d(
            kernel_size = 3,
            stride = 2,
            padding = 1 
        )  
        
        self.inception5a = InceptionBlock(
            in_channels = 832,
            out_1 = 256,
            out_2 = (160, 320),
            out_3 = (32, 128), 
            out_4 = 128 
        ) # -> 832 Channels
      
        self.inception5b = InceptionBlock(
            in_channels = 832,
            out_1 = 384,
            out_2 = (192, 384), 
            out_3 = (48, 128),
            out_4 = 128 
        ) # -> 1024 Channels
      
        self.avgpool = nn.AvgPool2d(kernel_size = 7, stride = 1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout( p = .4 ) 
        self.fc = nn.Linear(in_features = 1024, out_features = 1000) # -> Softmax(x) => pred
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.localresnorm1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.localresnorm2(x)
        x = self.maxpool2(x)
       
        x = self.inception3a(x)
        
        x = self.inception3b(x) 
        x = self.maxpool3(x)
        x = self.inception4a(x)
        aux1_x = self.aux1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2_x = self.aux2(x)
        x = self.inception4e(x)
        x = self.maxpool5(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
       
        x = self.avgpool(x)
        x = self.dropout(x)
        x = self.flatten(x) 
        x = self.fc(x)
        
        return x, aux1_x, aux2_x
        