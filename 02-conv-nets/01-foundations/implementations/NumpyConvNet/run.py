import numpy as np
from convnet import ConvNet

seed = 1 # define seed
np.random.seed(seed = seed) # set seed for reproducitbility

# TRAINING SET

# (B, C, H, W)
X_train = np.random.rand(10, 1, 28, 28).astype(np.float32)

# (10 samples, ) 3 classes!
y_train = np.random.randint(low = 0, high = 3, size = 10)

# TEST SET 

# (2 samples, 3, channels, 8 H, 8 W)
X_test = np.random.rand(2, 3, 8, 8).astype(np.float32)
# (2 samples, ) 3 classes!
y_test = np.random.randint(0, 3, 2)

# set hyperparams

layers = ['C', 'C', 'MP', 'FC', 'FC']
n_output_channels = [3, 2] # works
layer_size = [2, 2, 2, 32, 3] # works
activations = ['leakyrelu', 'leakyrelu', 'leakyrelu', 'softmax']
padding = [1, 2, 3 ] # works
stride = [1, 2, 3] # works
dilation_rate = [(1, 2), (2, 1)] 
param_init = 'glorot'

# init model and train

nn = ConvNet(seed = seed) # seed set
nn.train(
    
    X = X_train,
    y = y_train, 
    layers = layers, 
    n_output_channels = n_output_channels, 
    layer_size = layer_size,
    padding = padding,
    stride = stride,
    activations = activations,
    dilation_rate = dilation_rate,
    param_init = param_init
    
    )
