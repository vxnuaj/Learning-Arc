# MobileNet

> [Resource 1](https://www.coursera.org/learn/convolutional-neural-networks/lecture/9BqTk/mobilenet-architecture) | [Resource 2](https://www.coursera.org/learn/convolutional-neural-networks/lecture/B1kPZ/mobilenet) | [Resource 3](https://arxiv.org/abs/1704.04861v1)

In a Standard Conv, computational complexity is:

```math

\mathcal{O}(\mathcal{K}_H \times \mathcal{K}_W \times \mathcal{K}_{ch} \times X_{pos} \times \text{out}_{ch})

```

wehre $\mathcal{K}$ is the kernel and $X_{pos}$ is the total count of positions of the input, $X$, at which we apply a given convolution with $\mathcal{K}$. Given that we need multiple $\mathcal{K}$, we also account for $\text{out}_{ch}$ which can also be interpreted as the count of total $\mathcal{K}$.

To reduce computational complexity, we can use a Depthwise Seperable Convolution.

### DepthWise Conv

For an input $X$ of shape $(in_{ch} \times \text{H} \times \text{W})$, we can apply a Depthwise Convolution by using a unique $\mathcal{K}$ for every $i \in \text{Input Channels}$, to get an output $\hat{X}$, of size $(in_{ch} \times H_{out} \times W_{out})$.

Then the computational cost for a single depthwise convolution becomes:

```math

\mathcal{O}(\mathcal{K_H} \times \mathcal{K}_W \times X_{pos} \times {in}_{ch})

```

which is about $\frac{\mathcal{K}_{ch} \times out_{ch}}{\text{in}_{ch}}$ of a factor smaller than a regular convolution.

### Seperable ($1 \times 1$)

Then we can apply a pointwise convolution.

This lets us combine the seperately learnt features, into a single set of feature maps for the output, $Z$.

The convnet is essentially learning the relationships of features across the multiple output feature maps of the DepthWise Convolution.

Computational Complexity then becomes, for the entire Depthwise Seperable Convolution

```math

\mathcal{O}(\mathcal{K_H} \times \mathcal{K}_W \times X_{pos} \times {in}_{ch}) + \mathcal{O}(\hat{X}_{pos} \times \text{pointwise}_{ch} \times \text{in}_{ch})

```

which becomes considerably cheaper, as a whole.

### MobileNet

- The original MobileNet paper uses the above DepthWise Seperable Convolution, 13 times
- MobileNet V2 uses a residual connection, such that the "MobileBlock", includes an expansion layer ($1 \times 1$ conv, to expand the count of input feature maps to the Depthwise Conv.) and then a Depthwise Seperable Convolution, with a residual connection from the input to output. This was used $17$ times.
  
   > Note, while a $1 \times 1$ convolution is typically used to downsample before a larger convolution, it seems that we want to expand the set of feature maps for the depthwise conv. [TODO] -- why? i conjecture it's to ensure that the depthwise conv is able to learn the appropriate count of features, while still being memory efficient (though tuning the channel count of the $1 \times 1$ expansion layer would be needed to ensure an optima.)

## MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications

### Abstract & Introduction

- MobileNets use Depthwise Seperable COnvs to build lightweight deep Neural Networks to run on edge devices.
- 2 sets of hyperparamters allow for the model to be constructed to the right size given the computational limits of the given device the model will be operating on.

### Priorwork

- MobileNet focuses on both *size* and *speed*, which proves to be an outlier as other networks purely focused on model *size*.

### MobileNet Architecture

- The model is based on DepthWise Seperable Convolutions
- A Standard Convolution combines the input input channels into as ingle channel, in a single step. The Depthwise convolution splits this into 2 separate layers, via a Depthwise conv an then a pointwise convolution. 
- Depthwise Convs introduce a single Kernel for each channel, such that computational cost reduces by a factor of $M$, where $M$ is the count of channels in a $\mathcal{K}$ for a regular convolution. 
  - But it doesn't combine features across channels, we're only learning representations for individual feature maps -- we need recombination via $1 \times 1$ convolutions

<br/>

- The architecture is built on Depthwise Seperable COnvs besides the first layer which is a full convolution.
  - Each layer goes as : $\text{Conv} \rightarrow \text{BatchNorm} \rightarrow \text{ReLU}$
  - Downsampling is handled by strided convolutions for in the Depthwise Sep. Conv. Layers, $s = 2$     
    - In the first layer as well, handled via the first $3 \times 3$ conv.

<br/> 

- It is'nt enough to purely consider model capacity as a measure for computational cost, ensuring that the model has **fast** inference is also important.
  - Given that MobileNet uses $1 \times 1$ convolutions for most of it's computation, it can be turnt into a $\text{GEMM}$ fairly easily.

- We can introduce a width mulitplier, $\alpha$, which thins the size of MobileNet, such that for any number of input and output channels, $M$ and $N$ respsectively, the input and output channels become $\alpha M$ and $\alpha N$.
  - Done prior to training -- expecting accuracy to decrease for smaller $\alpha$
  - $\alpha \in [0, 1)$
  - This has the effect of automating the proportional scaling down each layer, rather than manually adjusting each layer, in a disproportionate manner
  
- Can also introduce the resolution multiplier, $\rho$, which reduces the spatial dimensions of a given set of input feature maps to the $\ell th$ layer by a factor of $\rho$, where $\rho \in [0, 1)$  
  - only applied to the input layer -- for the original image. 
  - reduces computational cost by $\rho^2$

### Experiments

- Compared to a regular ConvNet, the MobileNet (Depthwise Seperable Convs) architecture only gets $1$% worse compared to the ConvNet, on ImageNet, with a huge amount in Param decrease (only having $11$% of of ops and $14$% of parameters)