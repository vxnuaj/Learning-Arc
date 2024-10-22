import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F

class VGG11(nn.Module):
    
    '''
    
    Input: 3 x 224 x 224 Images | (B, C, H, W)
    
    ''' 
    
    def __init__(self):
        super().__init__()
       
        self.vgg1 = nn.Sequential(
                    
                    nn.Conv2d(kernel_size = 3, in_channels = 3, out_channels = 64, padding = 'same'), # -> 64 x 224 x 224
                    nn.MaxPool2d(kernel_size=2, stride = 2) # -> 64 x 112 x 112
                    
        )
        
        self.vgg2 = nn.Sequential(
                    
                    nn.Conv2d(kernel_size = 3, in_channels = 64, out_channels = 128, padding = 'same'), # 128 x 112 x 112
                    nn.MaxPool2d(kernel_size = 2, stride = 2) # 128 x 56 x 56

        )

        self.vgg3 = nn.Sequential(
            
            nn.Conv2d(kernel_size = 3, in_channels = 128, out_channels = 256, padding = 'same'), # 256 x 56 x 56
            nn.ReLU(),
            nn.Conv2d(kernel_size = 3, in_channels = 256, out_channels = 256, padding = 'same'), # 256 x 56 x 56
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2) # 256 x 28 x 28
        )
        
        self.vgg4 = nn.Sequential(
            
            nn.Conv2d(kernel_size = 3, in_channels = 256, out_channels = 512, padding = 'same'), # 512 x 28 x 28
            nn.ReLU(),
            nn.Conv2d(kernel_size = 3, in_channels = 512, out_channels = 512, padding = 'same'), # 512 x 28 x 28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2) # 512 x 14 x 14
            
        )
        
        self.vgg5 = nn.Sequential(
            
            nn.Conv2d(kernel_size = 3, in_channels = 512, out_channels = 512,  padding = 'same'), # 512 x 14 x 14
            nn.ReLU(),
            nn.Conv2d(kernel_size = 3, in_channels = 512, out_channels = 512, padding = 'same'),  # 512 x 14 x 14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),  # 512 x 7 x 7
            nn.Flatten() # -> 25088
        )
       
        self.fc1 = nn.Linear(in_features = 512 * 7 * 7, out_features = 4096)
        self.dropout1 = nn.Dropout(p = .5)
        self.fc2 = nn.Linear(in_features = 4096, out_features = 4096)
        self.dropout2 = nn.Dropout(p = .5)
        self.fc3 = nn.Linear(in_features = 4096, out_features=1000) 
       

    def forward(self, x):

        x = self.vgg1(x)
        x = self.vgg2(x)
        x = self.vgg3(x)
        x = self.vgg4(x) 
        x = self.vgg5(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)

        return x 


class VGG13(nn.Module):
    
    '''
    
    Input: 3 x 224 x 224 Images | (B, C, H, W)
    
    ''' 
    
    def __init__(self):
        super().__init__()
       
        self.vgg1 = nn.Sequential(
                    
                    nn.Conv2d(kernel_size = 3, in_channels = 3, out_channels = 64, padding = 'same'), # -> 64 x 224 x 224
                    nn.Conv2d(kernel_size = 3, in_channels = 64, out_channels = 64, padding = 'same'), # -> 64 x 224 x 224
                    nn.MaxPool2d(kernel_size=2, stride = 2) # -> 64 x 112 x 112
                    
        )
        
        self.vgg2 = nn.Sequential(
                    
                    nn.Conv2d(kernel_size = 3, in_channels = 64, out_channels = 128, padding = 'same'), # 128 x 112 x 112
                    nn.Conv2d(kernel_size = 3, in_channels = 128, out_channels = 128, padding = 'same'), # 128 x 112 x 112
                    nn.MaxPool2d(kernel_size = 2, stride = 2) # 128 x 56 x 56

        )

        self.vgg3 = nn.Sequential(
            
            nn.Conv2d(kernel_size = 3, in_channels = 128, out_channels = 256, padding = 'same'), # 256 x 56 x 56
            nn.ReLU(),
            nn.Conv2d(kernel_size = 3, in_channels = 256, out_channels = 256, padding = 'same'), # 256 x 56 x 56
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2) # 256 x 28 x 28
        )
        
        self.vgg4 = nn.Sequential(
            
            nn.Conv2d(kernel_size = 3, in_channels = 256, out_channels = 512, padding = 'same'), # 512 x 28 x 28
            nn.ReLU(),
            nn.Conv2d(kernel_size = 3, in_channels = 512, out_channels = 512, padding = 'same'), # 512 x 28 x 28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2) # 512 x 14 x 14
            
        )
        
        self.vgg5 = nn.Sequential(
            
            nn.Conv2d(kernel_size = 3, in_channels = 512, out_channels = 512,  padding = 'same'), # 512 x 14 x 14
            nn.ReLU(),
            nn.Conv2d(kernel_size = 3, in_channels = 512, out_channels = 512, padding = 'same'),  # 512 x 14 x 14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),  # 512 x 7 x 7
            nn.Flatten() # -> 25088
        )
        
        self.fc1 = nn.Linear(in_features = 512 * 7 * 7, out_features = 4096)
        self.dropout1 = nn.Dropout(p = .5)
        self.fc2 = nn.Linear(in_features = 4096, out_features = 4096)
        self.dropout2 = nn.Dropout(p = .5)
        self.fc3 = nn.Linear(in_features = 4096, out_features=1000) 
       

    def forward(self, x):

        x = self.vgg1(x)
        x = self.vgg2(x)
        x = self.vgg3(x)
        x = self.vgg4(x) 
        x = self.vgg5(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)

        return x 

    
class VGG16(nn.Module):
    
    '''
    
    Input: 3 x 224 x 224 Images | (B, C, H, W)
    
    ''' 
    
    def __init__(self):
        super().__init__()
       
        self.vgg1 = nn.Sequential(
                    
                    nn.Conv2d(kernel_size = 3, in_channels = 3, out_channels = 64, padding = 'same'), # -> 64 x 224 x 224
                    nn.Conv2d(kernel_size = 3, in_channels = 64, out_channels = 64, padding = 'same'), # -> 64 x 224 x 224
                    nn.MaxPool2d(kernel_size=2, stride = 2) # -> 64 x 112 x 112
                    
        )
        
        self.vgg2 = nn.Sequential(
                    
                    nn.Conv2d(kernel_size = 3, in_channels = 64, out_channels = 128, padding = 'same'), # 128 x 112 x 112
                    nn.Conv2d(kernel_size = 3, in_channels = 128, out_channels = 128, padding = 'same'), # 128 x 112 x 112
                    nn.MaxPool2d(kernel_size = 2, stride = 2) # 128 x 56 x 56

        )

        self.vgg3 = nn.Sequential(
            
            nn.Conv2d(kernel_size = 3, in_channels = 128, out_channels = 256, padding = 'same'), # 256 x 56 x 56
            nn.ReLU(),
            nn.Conv2d(kernel_size = 3, in_channels = 256, out_channels = 256, padding = 'same'), # 256 x 56 x 56
            nn.ReLU(),
            nn.Conv2d(kernel_size = 3, in_channels = 256, out_channels = 256, padding = 'same'), # 256 x 56 x 56 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2) # 256 x 28 x 28
        )
        
        self.vgg4 = nn.Sequential(
            
            nn.Conv2d(kernel_size = 3, in_channels = 256, out_channels = 512, padding = 'same'), # 512 x 28 x 28
            nn.ReLU(),
            nn.Conv2d(kernel_size = 3, in_channels = 512, out_channels = 512, padding = 'same'), # 512 x 28 x 28
            nn.ReLU(),
            nn.Conv2d(kernel_size = 3, in_channels = 512, out_channels = 512, padding = 'same'), # 512 x 28 x 28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2) # 512 x 14 x 14
            
        )
        
        self.vgg5 = nn.Sequential(
            
            nn.Conv2d(kernel_size = 3, in_channels = 512, out_channels = 512,  padding = 'same'), # 512 x 14 x 14
            nn.ReLU(),
            nn.Conv2d(kernel_size = 3, in_channels = 512, out_channels = 512, padding = 'same'),  # 512 x 14 x 14
            nn.ReLU(),
            nn.Conv2d(kernel_size = 3, in_channels = 512, out_channels = 512, padding = 'same'),  # 512 x 14 x 14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),  # 512 x 7 x 7
            nn.Flatten() # -> 25088            
        )
        
        self.fc1 = nn.Linear(in_features = 512 * 7 * 7, out_features = 4096)
        self.dropout1 = nn.Dropout(p = .5)
        self.fc2 = nn.Linear(in_features = 4096, out_features = 4096)
        self.dropout2 = nn.Dropout(p = .5)
        self.fc3 = nn.Linear(in_features = 4096, out_features=1000) 
       

    def forward(self, x):

        x = self.vgg1(x)
        x = self.vgg2(x)
        x = self.vgg3(x)
        x = self.vgg4(x) 
        x = self.vgg5(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)

        return x 
    
class VGG19(nn.Module):
    
    '''
    
    Input: 3 x 224 x 224 Images | (B, C, H, W)
    
    ''' 
    
    def __init__(self):
        super().__init__()
       
        self.vgg1 = nn.Sequential(
                    
                    nn.Conv2d(kernel_size = 3, in_channels = 3, out_channels = 64, padding = 'same'), # -> 64 x 224 x 224
                    nn.Conv2d(kernel_size = 3, in_channels = 64, out_channels = 64, padding = 'same'), # -> 64 x 224 x 224
                    nn.MaxPool2d(kernel_size=2, stride = 2) # -> 64 x 112 x 112
                    
        )
        
        self.vgg2 = nn.Sequential(
                    
                    nn.Conv2d(kernel_size = 3, in_channels = 64, out_channels = 128, padding = 'same'), # 128 x 112 x 112
                    nn.Conv2d(kernel_size = 3, in_channels = 128, out_channels = 128, padding = 'same'), # 128 x 112 x 112
                    nn.MaxPool2d(kernel_size = 2, stride = 2) # 128 x 56 x 56

        )

        self.vgg3 = nn.Sequential(
            
            nn.Conv2d(kernel_size = 3, in_channels = 128, out_channels = 256, padding = 'same'), # 256 x 56 x 56
            nn.ReLU(),
            nn.Conv2d(kernel_size = 3, in_channels = 256, out_channels = 256, padding = 'same'), # 256 x 56 x 56
            nn.ReLU(),
            nn.Conv2d(kernel_size = 3, in_channels = 256, out_channels = 256, padding = 'same'), # 256 x 56 x 56 
            nn.ReLU(),
            nn.Conv2d(kernel_size = 3, in_channels = 256, out_channels = 256, padding = 'same'), # 256 x 56 x 56 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2) # 256 x 28 x 28
        )
        
        self.vgg4 = nn.Sequential(
            
            nn.Conv2d(kernel_size = 3, in_channels = 256, out_channels = 512, padding = 'same'), # 512 x 28 x 28
            nn.ReLU(),
            nn.Conv2d(kernel_size = 3, in_channels = 512, out_channels = 512, padding = 'same'), # 512 x 28 x 28
            nn.ReLU(),
            nn.Conv2d(kernel_size = 3, in_channels = 512, out_channels = 512, padding = 'same'), # 512 x 28 x 28
            nn.ReLU(),
            nn.Conv2d(kernel_size = 3, in_channels = 512, out_channels = 512, padding = 'same'), # 512 x 28 x 28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2) # 512 x 14 x 14
            
        )
        
        self.vgg5 = nn.Sequential(
            
            nn.Conv2d(kernel_size = 3, in_channels = 512, out_channels = 512,  padding = 'same'), # 512 x 14 x 14
            nn.ReLU(),
            nn.Conv2d(kernel_size = 3, in_channels = 512, out_channels = 512, padding = 'same'),  # 512 x 14 x 14
            nn.ReLU(),
            nn.Conv2d(kernel_size = 3, in_channels = 512, out_channels = 512, padding = 'same'),  # 512 x 14 x 14
            nn.ReLU(),
            nn.Conv2d(kernel_size = 3, in_channels = 512, out_channels = 512, padding = 'same'),  # 512 x 14 x 14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),  # 512 x 7 x 7
            nn.Flatten() # -> 25088
            
        )
        
        self.fc1 = nn.Linear(in_features = 512 * 7 * 7, out_features = 4096)
        self.dropout1 = nn.Dropout(p = .5)
        self.fc2 = nn.Linear(in_features = 4096, out_features = 4096)
        self.dropout2 = nn.Dropout(p = .5)
        self.fc3 = nn.Linear(in_features = 4096, out_features=1000) 
       

    def forward(self, x):

        x = self.vgg1(x)
        x = self.vgg2(x)
        x = self.vgg3(x)
        x = self.vgg4(x) 
        x = self.vgg5(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)

        return x 
        