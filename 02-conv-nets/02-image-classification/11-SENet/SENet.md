# Squeeze-and-Excitation Networks

### Initial Thoughts

Seems like their gating mechanism with $\sigma$, then element wise multiplied with the original output, resembles attention -- the greater the output of the $\odot$ operation, the more attention subsequent layers pay attention to the given feature map, $Z$. 

Interesting...


### Abstract

Prior with into CNNs has focused on improving the quality of spatial features on each feature map.

Instead, they focus on improving the channel-wise representations by the *"Squeeze-and-Excitation"* (SE) block, which adaptively recalibrates channel responses by modelling relationships between channels.

Won first palce in Image Net 2017 for Classification.

### Introdution

The theme of computer vision is the search for more powerful feature representatiosn that capture the most important properties of the image for the given task, to enable improved performance in classification.

> Don't forget, ConvNets are purely feature extractors -- the final classification being done by the final layer ($1 \times 1$ convolution or fc, into softmax).

More so, modern architectures like Inception focus on improving spatial dependencies -- in this paper the focus is on relationships between channels, or feature maps.

For any given transformation, $\textbf{F}$, which maps $\text{X} \rightarrow \text{U}$, where $U \in \mathbb{R}^{H \times W \times C}$, you can construct a Squeeze and Excitation Block that recalibraates the features.

$U$ is passed through a *squeeze* operation, which outputs a "channel descriptor". The channel descriptor produces an embedding representing the global distribution of channel-wise features.

Then the *excitation* operation uses a set of parameters as a gating mechanism, to modulate the magnitude of the features that will be output into the next layer.

The role of SEBlocks, despite the generic constructions, plays different roles at different depths of a given network.

1. In earlier layers, SEBlocks perform *excitation* in a class-agnostic manner -- that is extracting general and important features (corners, edges, etc) without having too much information about the correct class.

2. In later layers, SEBlocks perform *excitation* in a more class specific manner, such that their *excitation* selectively amplifies the feature maps that have more of a role for the final classification.

### Squeeze-and-Excitation Blocks

A convolution behaves as the element wise multiplication of $\mathcal{K}$ with $X_{i:\mathcal{K_r}, j:\mathcal{K_c}}$ and then summation, across all channels of $X$, to yield a single output, $\hat{X}$.

The issue is that $\mathcal{K}$, while being able to learn spatial dependencies, channel-wise dependencies aren't explicitly modelled and are of second thought to the modelling of $\mathcal{K}$.

While there might be some channel influence to a given parameter of $\mathcal{K}$, note that the output to a single convolution operation, is a single channel -- such that spatial information gets tangled with channel-wise information -- due to the summation.

To enable learning channel interdependencies, they construct the SEblock:

1. Global Average Pool the input, $U$, where $U$ is the output to a convolution
2. Feed the output of the global average pooling into an fully connected layer -> ReLU -> FC -> Sigmoid, to get a set of activations, $s$, same count as the channels in the input.
3. Element wise multiply $s$ w.r.t to the channel dimension

> Note that the capcity of the initial FC layer is $\frac{C}{r}$ and the capacity of the second is $C$ -- we're reducing dimensionality and then expanding back to the original size of $C$.

```math

z = \text{GlobalAvgPool | }(U) (c \times 1 \times 1) 
\\[3mm]
s = \sigma(W_2\text{ReLU}(W_1z)) | (c \times 1 \times 1)
\\[3mm]
\hat{U} = U \odot s | (c \times n \times m)

```

where the input is of dimension $c \times n \times m$

Essentially, we're transforming $z$ via a set of learnt channel weights -- done via global average pooling -> fc layers, such that the output $U$ is scaled via the transformed $z$, scaling down less relevant features.

