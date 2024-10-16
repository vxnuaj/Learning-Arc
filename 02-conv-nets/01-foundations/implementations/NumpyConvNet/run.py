import numpy as np
from convnet import ConvNet

seed = 1
np.random.seed(seed)

def generate_shape(label, size=3):
    img = np.zeros((size, size))
    if label == 0:  # Square
        img[0:2, 0:2] = 1
    elif label == 1:  # Circle
        y, x = np.ogrid[:size, :size]
        mask = (x - size//2)**2 + (y - size//2)**2 <= (size//3)**2
        img[mask] = 1
    return img

X_train = np.array([generate_shape(0), generate_shape(1), generate_shape(0), generate_shape(1)])  
X_train = X_train[:, np.newaxis, :, :] 
y_train = np.array([0, 1, 0, 1])  

X_test = np.array([generate_shape(0), generate_shape(1)])  
X_test = X_test[:, np.newaxis, :, :]  
y_test = np.array([0, 1])  


layers = ['C', 'C', 'FC', 'FC']  
n_output_channels = [1, 1]      
layer_size = [3, 3, 3, 2]       
activations = ['leakyrelu', 'leakyrelu', 'leakyrelu', 'softmax']  
padding = [1, 1]                
stride = [1, 1]                  
dilation_rate = [(1, 1), (1, 1)]
param_init = 'glorot'

nn = ConvNet(seed=seed)

nn.train(
    X=X_train,
    y=y_train,
    layers=layers,
    n_output_channels=n_output_channels,
    layer_size=layer_size,
    padding=padding,
    stride=stride,
    activations=activations,
    dilation_rate=dilation_rate,
    param_init=param_init,
    alpha=0.01, 
    epochs=1000
)
