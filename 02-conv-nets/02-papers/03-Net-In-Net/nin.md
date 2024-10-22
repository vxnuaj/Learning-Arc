# Network in Network

The issue with the previous LeNet, AlexNet, and VGGNet architectures were that

1. Their FC layers consume a tremendous amount of parameters (400MB for VGG WEights at FC Layers)
   1. No ability to work on smaller devices such as phones
2. Cannot add FC layers in the middle of a convolutional block, as we'd destroy spatial structure and require more memory.

Insted NiN uses global averagep ooling and $1 \times 1$ convolutions to mitigate these issues.

> global average pooling reduces a given $n \times n$ feature map to a single averaged value across all $n$. This integrates all information while reducing dimensionality, you reduce less when compard to global max pooling.

$1 \times 1$ convolutions replace the FC layers, while still being in the middle of the convolution block.

Essentially, for each $1 \times 1$ $\mathcal{K}$, we're learning a set of weights (equivalent to size of $C_{in}$), such that we're introducing a what you might call a *channelwise parametric pooling*.
You're able to learn the relationship between each $n \times n$ feature map across $C$ channels, *point-wise*. The more $\mathcal{K}$ you have (output $C$), the more relationships between pointwise features
across $C_{in}$ dimensions you will be able to learn.

Though, for each $\mathcal{K}$, a $1 \times 1$ conv layer is forced to learn a more broader hypothesis of the spatial structure of the input feature maps, due to weight sharing -- we only have 1 $\mathcal{K}$ vector of size $C_{in}$ for a given input where in an FC layer, we'd have a $W$ of size $C_in \times H_in \times W_in$.

Compared to a fully connected layer, a $1 \times 1$ convolution layer less parameters, by about a factor of $H_{in} \times W_{in}$.

## NiN Paper

### **Introduction**

- A **trained** convolution filter in a ConvNet is essentially a generalized linear model (GLM), for the given input.
  - The output of a convolution layer without a non-linearity can be seen akin to linear regression, each output being able to construct a regression line. You're essentially constructing a regression line that represents the linear relationship between different spatial features in the input.
- The convolution can be replaced by sliding an $\text{MLP}$ across multiple areas over the input, this is the essence of $\text{NiN}$. Each output feat ure map is the globally average pooled, into a vector value that is suitable for softmax classification

### **Network in Network**

<div align = 'center'>
<img src = 'https://vitalab.github.io/article/images/nin/sc01.jpg' width = 500>
<br>
<img src = 'https://miro.medium.com/v2/resize:fit:1400/1*PoNVgeyx6_KWR8vY083VdA.png' width = 500>
</div>
<br>

- For a given mlpconv layer, the MLP with $\text{ReLU}$ is used as a replacement for the GLM ($\mathcal{K}$), to convolve over each layer.

```math

f_{i, j, k}^1 = max(w_k^Tx_{i, j} + b_{k}, 0)
\\[3mm]
f_{i, j, k}^n = max(w_k^T f_{i, j}^{n - 1} + b_{k}, 0)

```

- where $x$ is the input feature map and $f_{i, j}^{n - 1}$ is the output of the previous layer in mlpconv.

- For softmax classification layer, the input should be of $n$ feature maps where $n = \text{Num of Classes}$. Each feature map is then globally average pooled to summarize the extracted features. The resulting feature vector is then input into the Softmax layer for a probability output.
  - An advantage is no additional parameters for Softmax layer.
  - Introduces more invariance to shifts in the input image, as we're summarizing ALL info from a feature map, removing the importance of spatial positioning.
  
- Networks used for testing consisted of three stacked mlpconv layers, followed by a max-pooling layer to sample the input iamge down by a factor of two ($3 \times 3, s = 2$)
- Each mlpconv layer, can essentially be seen as as $1 \times 1$ convolution, enabling the "depth-wise parametric pooling"
- $\Theta$'s initialized as $\sim \mathcal{N}(0, .01)$ while $\beta = 0$
- Dropout is applied on all outputs of mlpconv layers besides the output.
- Weight decay!