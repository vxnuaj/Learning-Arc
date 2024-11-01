# ResNext

*An extenstion of ResNet, this time using parallelized grouped convolutions, similar to Inception, but the convolutions are symmetric across their branches / groups*

**Initial Thoughts**

Given that sequential processing of a Neural Network can lead to slower learning and a narrow set of learned features, parallelizing convolutions makes sense as you have multiple unique $\mathcal{K}$, each applied onto seperate divided groups of the input, such that you learn different features which are more diversified.

This also makes sense as I decrease the amount of convolution ops, while **still increasing diversification of learned features**

Considering the following,

- 64 Input Channels
- 8 Groups / 8 Channels per Groups
- Each Group has 10 Output Channels
- Total is 80 Output Channels for this Residual Block.

```math

\mathcal{O}(8 \cdot 10 \cdot 8) = \mathcal{O}(640) = 640 \text{ convops}

```

If I did purely:

- 64 Input Channels
- 80 Output Channels

without Groups, then,

```math

\mathcal{O}(8\cdot 64) = \mathcal{O}(5120) = 5120 \text{ convops}

```

### Introduction

- Given inception modules, their construction of using lower dimensional embeddings via smaller $\mathcal{K} \in \mathbb{R}^n$ within separate branches, to represent / replace a larger $\mathcal{K}$, call it $\hat{\mathcal{K}} \in \mathbb{R}^m$, shows that the solution space is an $\mathbb{R}^n$ subspace of $\mathbb{R}^m$.

- The important property of this is high accuracy with a lower computational cost relative to models like VGG.
- But Inception is highly specialized, such that it might not be straightforward to adapt the architecture to other tasks.
- We introduce an architecture, that adapts VGG / ResNet of repeating layers while using the multi-branched paralellized convolutions by Inception
  - Their branched convolutions are of same dimension for all other parallel branch, such that their modules don't need to be high specialized and can be generalized across multiple scenarios
- They outperform ResNet, while still maintaiing computational complexity and model size.

### Related Work

- Grouped Convolutions can date back to AlexNet, where parallelizing ALexNet on 2 GPUs isolated several $\mathcal{K}$ from one another, such that they learnt distinct features. It isn't new.
- The extrema to introducing cardinality is a Depthwise (a.k.a channelwise) convolution, such that $\text{cardinality} = 1$, each input channel as a separate $\mathcal{K}$. Of course, to increase expressiveness of the network after each depthwise convolution, we can use multiple $1 \times 1$ convolutions to aggregate the features such that we learn the relationships.

### Method

- A design inspired from VGG and ResNets is adopted.
- Their residual connections have the same structure as ResNet and is inspired by the rules followed by VGG / ResNets
  - If over several layers, the output size remains the same, then for $\mathcal{K}$, we maintain the same hyper-params -- indicating same width and filter size.
    - Eliminating complexity of constructing a "specialized" block, unlike shown in Inception.
  - If a spatial map is dowmsampled by a factor of $2$, the widths of the network is multipled by $2$ -- half the size, twice the output feature maps.
    - Ensures computational complexity is the same for all blocks.

**3.2**

Revisiting simple neurons, where we have $\sum_{i = 1}^D w_i x_i$, they introduce $\sum^C_{i = 1} \tau_i(x)$, where the transformation $\tau$ is applied onto a group of $x$, projecting it into a lower dimensional space and the transforming it (referring to depthwise seperable convolutions with varying degrees of cardinality, $C$)

> Apply the convolution operation for all $g$ groups, up to $C$ (cardionality), then recombine via $1 \times 1$ convolutions. This is the essence of their version of depthwise seperable convs, via $\tau$.

- They set all $\tau$ to be of the same topology for simplicity.

They construct $\tau$ as:

<div align = 'center'>
<img src = 'resnextblock.png' width = 700>
</div>

left off on beginning of page 4