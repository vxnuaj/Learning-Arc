import numpy as np
from convnet import ConvNet

# Set random seed for reproducibility
seed = 1
np.random.seed(seed)

# Function to generate simple shapes: 0 for square, 1 for circle
def generate_shape(label, size=3):
    img = np.zeros((size, size))
    if label == 0:  # Square
        img[0:2, 0:2] = 1
    elif label == 1:  # Circle
        y, x = np.ogrid[:size, :size]
        mask = (x - size//2)**2 + (y - size//2)**2 <= (size//3)**2
        img[mask] = 1
    return img

# TRAINING SET (Simple binary classification with square and circle patterns)
X_train = np.array([generate_shape(0), generate_shape(1), generate_shape(0), generate_shape(1)])  # 4 samples with 3x3 images
X_train = X_train[:, np.newaxis, :, :]  # Add channel dimension (C=1) at the correct position
y_train = np.array([0, 1, 0, 1])  # Labels: 0 (square) and 1 (circle)

# TEST SET (Simple binary test data)
X_test = np.array([generate_shape(0), generate_shape(1)])  # Two test samples
X_test = X_test[:, np.newaxis, :, :]  # Add channel dimension (C=1) at the correct position
y_test = np.array([0, 1])  # Labels: 0 (square) and 1 (circle)

# Verify the shapes
print("Training data shape:", X_train.shape)  # Expected: (4, 1, 3, 3)
print("Test data shape:", X_test.shape)  # Expected: (2, 1, 3, 3)

# Hyperparameters
layers = ['C', 'C', 'FC', 'FC']  # Two convolutional layers followed by fully connected layers
n_output_channels = [1, 1]       # One output channel for each conv layer
layer_size = [3, 3, 3, 2]        # 3x3 kernel size, 3 units for FC layers, 2 units in the last FC layer (binary output)
activations = ['leakyrelu', 'leakyrelu', 'leakyrelu', 'softmax']  # LeakyReLU activations followed by softmax
padding = [1, 1]                 # Padding to maintain the size after convolution
stride = [1, 1]                  # Stride of 1 for both convolution layers
dilation_rate = [(1, 1), (1, 1)] # No dilation
param_init = 'glorot'

# Initialize and train the model
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
    alpha=0.01,  # Reduced learning rate
    epochs=1000
)
