import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F

class AlexNet(nn.Module):
  
    '''
   
    Inputs are 3 X 224 X 224 RGB Images. 

    Train with:
    
    epochs = 50, lr = .01, beta = .9, decay_rate = .0005, p = .5
    '''
    
    def __init__(self):
        super().__init__()
     
        # compute input / ouput sizes as:
        # \frac{input_size + 2(P) - kernel_size}{stride} + 1
        # P = padding, D = dilation 
       
        self.conv1 = nn.Conv2d(kernel_size = 11, in_channels = 3, out_channels = 96, stride = 4) # -> 96 X 54 X 54
        self.maxpool1 = nn.MaxPool2d(kernel_size = 3, stride = 2) # -> 96 X 26 X 26
       
        # NOTE 
        # 
        # Notice how we're decreasing the size of the feature maps (downsampling), yet increasing the count of the feature maps for the outputs.
        # You can interpret this as extracting a smaller set of hierarchically important features, for the given receptive field of the model.
     
        self.conv2 = nn.Conv2d(kernel_size = 5, in_channels = 96, out_channels = 256, stride = 1, padding = 2) # -> 256 X 26 X 26
        self.maxpool2 = nn.MaxPool2d(kernel_size = 3, stride = 2)  # - > 256 X 12 X 12
        self.conv3 = nn.Conv2d(kernel_size = 3, in_channels = 256, out_channels = 384, padding = 1) # -> 384 X 12 X 12
        self.conv4 = nn.Conv2d(kernel_size = 3, in_channels = 384, out_channels = 256, padding = 1) # -> 256 X 12 X 12
        self.maxpool3 = nn.MaxPool2d(kernel_size = 3, stride = 2) # -> 256 X 5 X 5

        self.flatten = nn.Flatten() # -> B X 6400
        self.fc1 = nn.Linear(in_features=6400, out_features=4096) # -> B X 4096
        self.dropout1 = nn.Dropout(p = .5) 
        self.fc2 = nn.Linear(in_features=4096, out_features=4096) # -> B X 4096
        self.dropout2 = nn.Dropout(p = .5)
        self.fc3 = nn.Linear(in_features=4096, out_features=1000) # -> B X 1000
       
        
    def forward(self, x):
    
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = self.maxpool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.maxpool3(F.relu(self.conv4(x)))
        
        x = self.flatten(x)
        x = self.dropout1(self.fc1(x))
        x = self.dropout2(self.fc2(x))
        x = self.fc3(x)
       
        return x 