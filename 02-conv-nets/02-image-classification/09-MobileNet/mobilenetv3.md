# MobileNetV3

> [!NOTE] 
> Keep in mind, MobileNets aim to optimize ConvNets for edge devices, particularly mobile phones.

### Introduction

The goal of the paper is to develop the best computer vision architectures for mobile devices, meaning optimizing the accuracy and the latency trade off for devices with small computational power.

They

1. Develop Architecture Search Techniques
1. Implement efficient versions of nonlinearities for Mobile devices
2. Efficient network desitn
3. Efficient segmentation decoder.

### Related Work

Recent ConvNet architectures have not only been focused on decreasing parameter count, but also have been focused on decreasing inference time and computational complexity.

- SqueezeNet uses $1 \times 1$ convolutions to replace $3 \times 3$, while maintaining performance
- MobileNetV1 uses Depthwise seperable convs to decrease param count and increase inference speed by Depthwise Seperable Convs
- MobileNetV2 uses the Depthwise Seperable Convs, alongside an inverted residual block, to reduce the operating space of the model to $\mathbb{R}^m$ while expressiveness remains in the $\mathbb{R}^n$. This is done via an expansion $1 \times 1$ conv $\rightarrow$ $3 \times 3$ depthwise conv $\rightarrow$ linear $1 \times 1$ conv (given that a ReLU produces redundancy after $3 \times 3$ convolution).

### Efficient Mobile Building Blocks

MnasNet build upon the MobileNet architecture by introducing squeeze and excitation modules within the Inverted Residual BLock, allowing the output of the depthwise convolution to be weighted in an attention-like manner, prior to the linear $1 \times 1$ projection.

MobileNetV3 uses these modules, in the same structure, using the `swish` nonlinearity.

Swish is defined as:

```math

x \cdot \sigma(\beta x)

```

where $\sigma$ is the sigmoid activation, $(1 + e^{-x})^{-1}$, and $\beta$ is a trainable parameter via gradient descent or a hyperparameter.

but for MobileNet, the $\sigma$ function is changed to a hard $\sigma$, for more efficient computation.

### Network Improvements

<img width = 500 src = 'https://i0.wp.com/francescopochetti.com/wp-content/uploads/2020/08/2020-08-24_08h32_12-1.png?resize=1200%2C519&ssl=1'/>

<br/>

- MobileNet's second to last $1 \times 1$ convolution operates on a $7 \times 7$ feature map -- this is computationally redundant.
  - The model can be reduced to operating to a $1 \times 1$ set of feature maps, by moving the global average pooling layer to the prior layer.
  - They they observe that the $1 \times 1$ linear bottleneck isn't needed to reduce the channel-dimension for better computation (given the global avgpooling), so they remove the previous inverted convolution block alltogether.
    - with almost no reduction in accuracy.
- They're also able to reduce the initial layer to $16$ filters, rather than $32$, retaining the same accuracy.

They replace the use of the sigmoid function to it's hard piecewise version:

```math

\frac{\text{min(max(0, x + 3), 6)}}{6}
```

and then swish becomes

```math

\text{swish} = x(\frac{\text{min(max(0, x + 3), 6)}}{6})

```

There is no decrease in accuracy and the use of $\text{ReLU6}$ function allows for the mitigation of overflow in 8-bit CPU cores.

- hard-swish is only used in teh second half of the model, as it provides the most benefit there only. The rest of the layers interchangeably use $\text{ReLU}$

**The Squeeze and Excitation** reduction ratio was fixed to be $\frac{1}{4}$ of the number of channels in the expansion layer (or size of the input).

> $r = .25$