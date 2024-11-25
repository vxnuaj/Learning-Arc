# MobileNetV2: Inverted Residuals and Linear Bottlenecks

### Abstract

- MobileNet achieved SOTA for mobile models on mult. tasks / benchmark being capable of accurate semantic segmentation, while being an efficient model on mobile devices.
  - Apply their model to SSDLite and demonstrate semantic segmentation through *"Mobile DeepLabv3"*
- Based on inverted Residual Structure -- shortcut connections are between thin $1 \times 1$ conv layers. The expansion layer, ($3 \times 3$?) uses depthwise convolutions to filter features via non-linearity ($\text{ReLU-6}$)

### Introduction

The main contribution is the inverted residual with a linear bottleneck layer.

The module takes in a low-dimensional representation, which is then expanded into a high dimensional space -- which is then filtered via a lightweight depthwise convolution, and then projected back to the low dimensional space, with a linear convolution.

> I'm thinking they use $1 \times 1$ conv for the projection onto the high dimensional space, use the $3 \times 3$ conv for learning the features via $\text{ReLU}-6$, and then project back to the low dimensional space using another $1 \times 1$ conv.
> 
> I'm noticing, the use of $1 \times 1$ convolutions are pretty common for proejction onto high dimensional or low dimensional channel spaces, pretty interesting.

### Preliminaries and Intuition

Depthwise Convolutions are the key building block for efficient neural networks.

You factorize a convolution into:

- A grouped Convolution, where the groups is equal to the number of total channels (1 $\mathcal{K}$ per channel).
- Aftward, taking the independently learned features from each input channel, you learn the relationships between those through a pointwise, $1 \times 1$ convolution.

Standard Convs have a computational complexity of 

```math

\mathcal{O}(X_{\text{h}} \times X_{\text{w}} \times X_{\text{ch}} \times \mathcal{K}_{\text{ch}} \times \mathcal{K}_h \times \mathcal{K}_w)

```

while Depthwise Seperable Convs have a computational compelxity of:

```math

\mathcal{O}(X_{\text{h}} \times X_{\text{w}} \times X_{\text{ch}} \times \mathcal{K}_h \times \mathcal{K}_w) + \mathcal{O}(X_{\text{h}} \times X_{\text{w}} \times X_{\text{ch}} \times \mathcal{K}^{1 \times 1}_{\text{ch}})
```

Considering an input of size $3 \times 7 \times 7$, a standard convolution with $\mathcal{K}$ of shape $3 \times 3 \times 3$ (assuming $\text{stride} = 1$ and $\text{padding} = 0$, $\rightarrow$ $3 \times 5 \times 5$) would yield:

```math

3 \times 7 \times 7 \times 3 \times 3 \times 3 = 3,969 \text{ multi-adds}

```

while a depthwise seperable convolution, of same output size ($3 \times 5 \times 5$), would yield:

```math

(3 \times 7 \times 7 \times 3 \times 3) + (5 \times 5 \times 3 \times 3) = 1,323 + 225 = 1,548 \text{ multi-adds}

```

Clearly the depthwise seperable convolutions have a lower computational cost than a regular full-sized convolution, being $\text{256}\%$ more efficient, and empirically showing near same accuracy, albeit with a tiny bit of decreased performance.

Such convolution operations are used widely in MobileNets, Xception, and other models designed to run on edge devices.

Considering a Neural Network, $\mathcal{F}$, as a Manifold Learner, for a set of input feature maps, $X_i \in \mathbb{R}^n$, to layer $\ell_i \in L$, where $L = \set{\ell_1, \ell_2, \dots, \ell_n}$, the non-linear activation, $H$, after the convolution operation in $\ell_i$, tends to embed the weighted sum, $Z_i \in \mathbb{R}^n$, into a lower dimensional space, $\mathbb{R}^m$, where $\mathbb{R}^m$ is a subspace of $\mathbb{R}^n$.

Thereby, for a convolution with a kernel $\mathcal{K_i} \in \mathbb{R}^n$, we can easily reduce the dimensionality of $\mathcal{K}_i$ to $\mathbb{R}^m$, to reduce the computational cost, while still retaining the important features. Such can be the case, given the prior that $H$ embeds features in the subspace of $\mathbb{R}^m$.

You could reduce the dimensionality, through the width multiplier introduced in MobileNetV1 -- that is until your depth is of the same representation as underlying manifold. But the non-linear activation, $H = \text{ReLU}$, can break this property down.

Assume $H = \text{ReLU}$, where $\text{ReLU} = \text{max}(0, z)$.

For an input to $\text{ReLU}$, $Z_i$, if $Z_i > 0$, $\text{ReLU}(Z_i) = Z_i$, else $\text{ReLU}(Z_i) = 0$.

Therefore, after applying the non-linearity to the high dimensional structure $\in \mathbb{R}^n$, we'll lose information that is negative valued, while the information that is $> 0$ will pass through as the $I$ transformation, remaining a linear transformation (given by the prior convolution).

If the input to $\text{ReLU}$ has essential data lying on a lower dimensional manifold in $\mathbb{R}^n$, it's viable that $\text{ReLU}$ can learn extract meaningful features, while discarding of the irrelevance (negative valued inputs), while still remaining in $\mathbb{R}^n$.

Therefore you have redundancy, after $\text{ReLU}$, such that applying a with multiplier $k$, will still yield redundancy after $\text{ReLU}$.

We can use the fact that if the essential features (positive valued outputs to the convolution, them essentially being a linear combination between $X$ and $\mathcal{K}$ at position of $X$ at $i, j$, given that $\text{ReLU}$ performs the $I$ transformation for postive valued inputs) are preserved while non-essential features are zeroed out, resulting in the retained information of the output to $\text{ReLU}$ lying on a subspace $\in \mathbb{R}^n$ to rationalize the use of linear bottleneck layers, meaning $1 \times 1$ convolutions as depthwise parametric pooling, to embed the high dimensional features $\in \mathbb{R}^n \rightarrow \mathbb{R}^m$, instead of applying $\text{ReLU}$ to the raw output of the original convolution.

> For the inverted Residual Block presented in MobileNetV2, this is used in the latter 2 layers, while the former is a $1 \times 1$ expansion layer.  
> 
> You (1) expand the low dimensional representation to a high dimensional space, (2) learn the important features via the depthwise 3x3 conv and ReLU6, (3) then project the learnt features in the high dimensional space to the low dimensional space.

- Expansion factor is now referred to as $t$, the multiplicity of expansion from $k$ channels to $tk$ channels after the $1 \times 1$ expansion layer.

### Inverted Residuals

A regular residual block goes as:

```math
 
\text{BottleNeck} \rightarrow 3 \times 3 \rightarrow \text{Expansion} \rightarrow \text{Output}

```

where $\text{Output} = Z + X$, where $Z$ is the output of the expansion layer and $X$ is the input to the residual block.

In the case of MobileNet, their inverted residual blocks go as:

```math

\text{Expansion} \rightarrow 3 \times 3 \text{ Depthwise} \rightarrow \text{BottleNeck} \rightarrow \text{Output}

```

where $\text{Output} = X + Z$ (as prior). 

The inverted residual connection connnects the $\text{BottleNeck}_{\ell - 1}$ to the $\text{BottleNeck}_{\ell}$, via an element wise summation, instead of connecting $\text{Expansion}_{\ell - 1}$ to $\text{Expansion}_{\ell}$, as done in residual blocks.

This is inspired by the aformentioned, *"We can use the fact that if the essential features (positive valued outputs to the convolution, them essentially being a linear combination between $X$ and $\mathcal{K}$ at position of $X$ at $i, j$, given that $\text{ReLU}$ performs the $I$ transformation for postive valued inputs) are preserved while non-essential features are zeroed out, resulting in the retained information of the output to $\text{ReLU}$ lying on a subspace $\in \mathbb{R}^n$ to rationalize the use of linear bottleneck layers, meaning $1 \times 1$ convolutions as depthwise parametric pooling, to embed the high dimensional features $\in \mathbb{R}^n \rightarrow \mathbb{R}^m$, instead of applying $\text{ReLU}$ to the raw output of the original convolution."*.

Thereby, for outputs to a given inverted residual block, you'll end up with a lower dimensional feature space, $\mathbb{R}^m$, rather than the higher dimensional space

The alternative scenario would be a $1 \times 1$ conv, acting as a means to learn representations across feature maps, without reduction into a lower dimensional feature space, and then applying a non-linearity, which was shown to be spatially redundant, given that $\text{ReLU}$ cancels out features, resulting in important features lying on a lower dimensional subspace.

This allows the future computations to focus on purely the important features $\in \mathbb{R}^m$, **without** the redundant parameters that focus on sparse representations $\in \mathbb{R}^n$

> ($\mathbb{R}^m < \mathbb{R}^n$)

### Information Flow Interpretation

Inverted Residual Blocks provide a seperation between the i/o domains of the model -- the bottleneck layers are the building blocks, seen as the capacity of the network at each layer, while the depthwise convolution with the non-linear transformation can be seen as the expressiveness of the model.

> Given that important information lies on the $\mathbb{R}^m$ feature space (post bottleneck), while expansion to the $\mathbb{R}^n$ space is done to expose features and relationships that will be selectively extracted (important features) and eliminated (non-important) via the BottleNeck, to $\mathbb{R}^m$

- Regular convolutional blocks have their expressiveness and capacity tangled together.

### Architecture

<div align = 'center'>
<img width = 400 src = 'https://figures.semanticscholar.org/dd9cfe7124c734f5a6fc90227d541d3dbcd72ba4/5-Table2-1.png'/>
</div><br/>

$\rho$ multiplier (resolution) applied to original input prior to the convolution.

$\alpha$ multiplier (width) applied to each layer output
- Besides the inner $3 \times 3$ depthwise convolution of the inverted residual block.

$3 \times 3$ is the default size for $\mathcal{K}$

1. Initial 32x3x3 CH. Conv 
2. 17 inverted residual blocks.
   1. 1x1 Conv, taking in $k$ channels and outputting $tk$ channels where $t$ is the expansion factor.
   2. 3x3 Depthwise Conv, $s=s$ for first $n$ block, else for $n > 1$, then $s=1$ (see table 2, above).
   3. linear 1x1 Conv, outputting $k'$ channels.
3. 1x1 Conv with $1280$ output channels
4. Global Average Pooling
5. 1x1 conv

- ReLU 6 as non-linearity
  - Working on Mobile / Edge devices that use low-bit numerical representations need quantized activations. $\text{ReLU-6}$ yields an upper activation value of $6$, to reduce overflow errors (given that 8-bit systems can only represnet values between a range of 0-255 or -128 - 128)
- kernel size 3x3 as standard kernel size.
- Dropout and BatchNorm during Training
- Use Width and Resolution Multipliers as in MobileNetV1
  - If any multiplier is less than $1$, the width multiplier isn't applied to the last conv layer.
