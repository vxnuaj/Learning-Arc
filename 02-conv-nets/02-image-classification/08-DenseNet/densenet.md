# DenseNet

Recalling the Taylor Expansion:

```math

f(x) = x_0 + \frac{f'(x_0)}x + ... + \frac{f^{(n)}}{n!}x^n

```

it splits a given $f(x)$ into an expansion of multiple functions.

This is in the vein of ResNet.

DenseNet, is similar to the former and the latter in the sense that the model, $F$, is made up of an expansion of multiple functions, added together sequentially, as ResNet.

The difference is that instead of adding an input to a given layer, $f_l$, as $f_l(x) + x$, instead there's a channel-wise concatenation, $[f_l, x]$, within a DenseBlock.

This allows for each $l + 1$ layer in a DenseBlock, to learn a more diverse set of features than $l$.

DenseNet is made up of two parts.  

- DenseBlocks
- Transition Layers
  - Serve as the means to downsample the channel dimension ($1 \times 1$ Convs) and the spatial dimension (via AvgPooling, $\mathcal{K} = 2, \text{stride} = 2$)

### DenseBlocks

- Makes use of ResNetV2 Identity Skip Connection.
  - BatchNorm -> ReLU -> Conv2d
- A denseblock consists of multiple convolutions, each one having the same count of output channels.
- Each input to the convolution is concatenated to the output to the convolution, channel-wise.
- The number of convolutions in the DenseBlock determines the *growth rate* of the output channels, relative to the input (you'll have a total of $\text{n * out}_l$, output channels for the denseblock).

### Transition Layers

- A Transition Layer reduces the number of channels via $1 \times 1$ convs, and height / width via strided average pooling layers ($\mathcal{K} = 2 \times 2$). -- to reduce complexity of the model in regards to the channel dimension.

### Output Layers

- Global Average Pooling -> FC Layer.

Why do we use average pooling rather than max-pooling in the transition layer?

I'd say to more effectively summarize features, rather than being biased towards the loudest feature, in order to pass as much information as possible to the DenseBlock.

One of the advantages mentioned in the DenseNet paper is that its model parameters are smaller than those of ResNet. Why is this the case?

Likely the Transition Layers


## Densely Connected Convolutional Networks

### Introduction

- We propose a *densely connected network*, of which for the $L$th layer, every single preceding $L-n$ layer is connected to it.
- DenseNet requires less parameters than ResNets.
  - ResNets were shown to have redundant layers, as in Stochastic Depth Networks, you could easily drop layers of ResNet randomly and still achieve good results.
  - Thereby, a DenseNet can have less parameters, and given that it concatenates information from a previous layer to the next, within each DenseBlock, we don't need as many parameters to learn meaningful features.

### DenseNets

- ResNets are typically formulated as $x_{\ell} = H(x_{\ell}) + x_{\ell}$, with the advantage of having $x_{\ell}$ as an identity transformation which allows for more effective gradient flow to earlier layers.
- But the $I$ transformation for the given residual connection may impede information flow within the network, as it's a simple summation to $H$, not allowing feature re-use but rather purely learning the residuals.
- BottleNeck layers -- given that for a DenseBlock, there may be a significant amount of input features, at each layer in the DenseBlock, we can reduce to $4k$ feature maps via $1 \times 1$ convolutions and then to $k$ with $3 \times 3$ convolutions.
- Transition Layers -- given that at each DenseBlock, we want to reduce the spatial dimensions but increase the count of output channels, to allow for efficient concatenation at each layer, the input to the denseblock must match dimensionality in the channel-dimensions and $H \times W$ dimensions.
  - So we introduce a $1 \times 1$ convolution and then $2 \times 2$ AvgPooling with $s = 2$ to reduce the spatial and channel dimensions to match the dimensionality of the outputs in the denseblock.
  - This is implemented with BN-ReLU-Conv and is referred to as DenseNet-B
- To compress the model, they reduce the feature-maps at transition layers.
  - If a DenseBlock has $m$ feature maps as their output, the following transition layer generates $\theta m$ feature maps, where $\theta$ is the compression factor, $\in 0 < \theta ≤ 1$
  - This is referred as DenseNet-C
  - The model that puts compression at bottleneck and transition layers is DenseNet-BC.
  
### Discussion

- DenseNet has improved model compactness alongside improved gradient flow (via deep supervision), that allows for better learning while still having a compact model size.