import numpy as np
import matplotlib.pyplot as plt
from convnet import ConvNet

# generate dataset
def generate_shape(label, size=16):
    img = np.zeros((size, size))
    
    if label == 0:
        img[4:12, 4:12] = 1
    elif label == 1:
        y, x = np.ogrid[:size, :size]
        mask = (x - size//2)**2 + (y - size//2)**2 <= (size//4)**2
        img[mask] = 1
    elif label == 2:
        img[4:12, 4:12] = 1
        for i in range(6):
            img[i, 4 + i] = 0  
    elif label == 3:
        np.fill_diagonal(img, 1)
        np.fill_diagonal(np.fliplr(img), 1)
    elif label == 4:
        img[7:9, :] = 1
        img[:, 7:9] = 1
    
    return img

########################################################################################################################################

#### TRAINING SET

X_train = np.array([generate_shape(i % 5) for i in range(500)])  # 500 samples, 5 classes
X_train = X_train[:, np.newaxis, :, :]  # Adding channel dimension (1 channel)
y_train = np.array([i % 5 for i in range(500)])  # Labels 0-4 for 5 classes

#### TESTING SET

X_test = np.array([generate_shape(i % 5) for i in range(100)])  # 100 samples, 5 classes
X_test = X_test[:, np.newaxis, :, :]  # Add channel dimension (1 channel)
y_test = np.array([i % 5 for i in range(100)])  # Labels 0-4 for 5 classes

'''
# visualizing imgs
fig, axes = plt.subplots(1, 5, figsize=(15, 8))
axes = axes.flatten()

for i in range(5):
    axes[i].imshow(X_train[i, 0], cmap='gray')
    axes[i].set_title(f"Class {i}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()
'''

# hyperparams
seed = 1
np.random.seed(seed)

layers = ['C', 'C', 'FC', 'FC']  # 2 convolutional layers, 1 fully connected layer
n_output_channels = [2, 4]  # First layer with 16 filters, second layer with 32
layer_size = [3, 3, 128, 5]  # 3x3 kernels, 128 units in the fully connected layer, 5 classes
activations = ['leakyrelu', 'leakyrelu', 'leakyrelu', 'softmax']  # LeakyReLU for conv layers, Softmax for output
padding = [1, 1]  # Padding to preserve spatial dimensions
stride = [1, 1]   # Stride of 1
dilation_rate = [(1, 1), (1, 1)]  # No dilation
param_init = 'glorot'  # Xavier/Glorot initialization for weights

nn = ConvNet(seed=seed)

X_train = X_train[0:15, :].reshape(15, -1, 16, 16)  # A small batch of 15 images
Y_train = y_train[0:15]

X_test = X_test[0:10, :].reshape(10, -1, 16, 16)
Y_test = y_test[0:10]

print('############ TRAINING ############')
print()

nn.train(
    X=X_train,
    y=Y_train,
    layers=layers,
    n_output_channels=n_output_channels,
    layer_size=layer_size,
    padding=padding,
    stride=stride,
    activations=activations,
    dilation_rate=dilation_rate,
    param_init=param_init,
    alpha=0.1, 
    epochs=100
)

print()
print('############ TESTING #############')
print()

nn.test(
    X_test = X_test,
    y_test = Y_test    
    
)