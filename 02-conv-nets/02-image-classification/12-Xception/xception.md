# Xception: Deep Learning with Depthwise Separable Convolutions

### 1.1 The Inception Hypothesis

- The inception network holds the inductive bias, that at every input to a given inception module, $X$ has a set of spatial features $\text{H} \times \text{W}$, that aren't correlated with the channel-wise features, $\text{C}$.
- Thereby, the inception module applies $1 \times 1$ convs, to learn features across $\text{C}$, typically in a subspace of $\mathbb{R}^n$ where $n = n_{X} \times n_{X}$, as $\mathbb{R}^m$ and then consolidate the spatial features within $\mathbb{R}^m$ using $3 \times 3$ or $5 \times 5$ convolutions. 
    - Given that channel-wise and spatial features aren't correlated, it doesn't make sense to learn then via the same set of parameters $\in \mathcal{K}$'
- Then considering a simplified inception module, each with $3$ branches of $1 \times 1 \rightarrow 3 \times 3$ branches (this can be seen as a large $1 \times 1$ conv, followed by a grouped $3 \times 3$ convolution), the question to ask is what is the correct number of groups $g$ for the grouped convolution and it's total size?

> Reviewing grouped Convs:
>
> Given an input with $36$ channels and a grouped convolution where $g = 3$ and output channels $= 18$, then the input is divided into $3$ groups, $\frac{36}{3} = 12$, and on those $12$ channels, we convolve with a $\mathcal{K}$ of size $12 \times 6 \times n \times n$ and concatenate, $6 + 6 + 6 = 18 \text{ output channels}$

The difference between the depthwise seperable convolutions and the extreme version of the aforementioned inception module (where $g = \text{ out channels}_{1 \times 1}$, meaning each channel has a separate kernel.), is that the extreme version of the inception module performs the $1 \times 1$ conv prior to the $3 \times 3$ and holds a $\text{ReLU}$ after the first $1 \times 1$


### The Xception Architecture.

They propose a ConvNet architecture based entirely on Depthwise Seperable Convs -- they make the hypothesis that learning the cross-channel correlations and the spatial correlations via $\mathcal{K}$ should be decoupled -- from the viewpoint that an inception module can be seen as a grouped convolution, this convnet architecture can be seen as an extreme version, hence XCeption (extreme incpetion).
