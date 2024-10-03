# Foundations

Feed forward neural networks are very efficient for learning the function that represents your training samples, $x_i \in \mathcal{X}$.

But when training a neural network on large datasets where each sample has an extremely large dimension, $\mathbb{R}^n$, scaling up these neural networks becomes very inefficient due to the extremely large parameter count which then leads to the need for high amounts of computational power and can risk overfitting.

If you had a $1000 \times 1000$ RGB image, you'd end up with $3,000,000$ input features. Say your hidden layer had $1000$ neurons, you'd have about $3,000,000,000$ parameters **only** for the 1st hidden layer.

> Damn, 2x larger than GPT-3

A better way to train your model on your data is to construct a broader hypothesis of your features, using a smaller set of weights with **deeper** layers. 

This is the essence of Convolutions.

### Invariance and Equivariance

Shift invariant, meaning for the input image, $X$, to the model, $f$, you'll get the same output, $\hat{y}$, despite changes to the original input $X$, in it's position.

Say we take the input, $X = [1, 2, 3]$, as an input to model, $f$

```math

f(X) = \hat{y}

```

Then we take the input $X + c$ to the model,, where $c$ is any scalar, added element wise to $X$.

```math

f(X + c) = \hat{y}

```

we get the same output as the model, $f$, is shift invariant.

**Shift Equivariance**, on the other hand is the tendency for the model to output a $\hat{y}$ that is reflective of changes done to $X$.

Say we have the model $f$ and input $X + c$:

```math

f(X+ c ) = \hat{y} + c

```

the output of $\hat{y} + c$, changing exactly as the input changes.

$f$ is shift equivariant. 

The convolutional layers in CNNs are shift equivariant, given that you're sliding the same kernel over an input $X$ and $X+c$, for $X + c$, you'll get the same output as $X$, the difference being that certain components are shifted to different spatial positions in the output feature map

This is given that the kernels define a set of weights that are shared by the entire layer, through the convolution operation for **different** spatial regions of the input features.

Meanwhile **pooling** layers are **shift invariant**, up to a certain degree depending on the degree the **poooling** layer is downsampling. This is as the same regions of a pooling layer will end up reducing the input features to a similar set of values, at least in the case of max pooling when the highest value feature remains within the same region of the input feature map

> *[nice](https://www.youtube.com/watch?v=a4Quhf9NhMY&t=39s)*

### The Holy Convolution

You could define a fully connected layer as:

```math
\begin{split}\begin{aligned} \left[\mathbf{H}\right]_{i, j} &= [\mathbf{U}]_{i, j} + \sum_k \sum_l[\mathsf{W}]_{i, j, k, l}  [\mathbf{X}]_{k, l}\\ &=  [\mathbf{U}]_{i, j} +
\sum_a \sum_b [\mathsf{V}]_{i, j, a, b}  [\mathbf{X}]_{i+a, j+b}\end{aligned}\end{split}

```
<em> From d2l.ai</em>

where $W$ or $V$ are fourth order tensors and:

- $i$ denotes the $ith$ set of fiters in $V$, also the row position of $H$
- $j$ denotes the $jth$ filter in the $ith$ set of filters in $V$
- $a$ denotes the row position of the $jth$ filter.
- $b$ denotes the column position of the $jth$ filter.
- $i + a$ denotes the row position in $X$
- $j + b$ denotes the column position in $X$

Essentially, we're multiplying each feature in a feature map $X$, by each $a, b$ parameter in $V$ to get the $i, j$ weighted sum in $H$. $i \times j$ is the total number of neurons in the hidden layer, to ultimately get the output feature map $H$ with $i \times j$ total activations.

This is purely a fully connected layer, expressed in a manner when $X$ is not flattened but as a matrix operation.

Now,

Let $Z$ be the feature map of the weighted sum, $W$ be the weights, $u$ be the bias, and $X$ be the inputs for a given layer.

A convolution (considering only a single channel) can be defined as:

```math
Z_{ij} = u + \sum_a \sum_b W_{ab}X_{i+a, j+b}

```

where:

- $i$ denotes the spatial position of the feature map of $Z$, in terms of *height*, and $j$ denotes the spatial postion of $Z$ in terms of *width*.

*These indices can be seen as the indices of which we slide $W$ over $X$ in the convolution operation... where we apply $i \times j$ weighted sums to get $Z$*.

- $a$ denotes the row position of $W$ and $b$ denotes the column position of $W$
- $i + a$ denotes the row and $i + b$ denotes the column of $X$ at which we apply the weighted sum with $W$

Essentially, we're taking the weight matrix, $W$, and sliding it for all positions in $X$. For every position $i + a, j+b$ in $X$, we perform and element wise multiplication with $W$ and $X$ and then a summation over all values of the element-wise multiplied matrix, to ultimately to get the $i, j$ weighted sum in $Z$. 

Prior to outputting to $Z$, we perform an element wise addition with the bias $b$

<div align = 'center'>
<img src = 'https://upload.wikimedia.org/wikipedia/commons/0/04/Convolution_arithmetic_-_Padding_strides.gif' width = 300px>
</div>

In the convolution operation, rather than having different $i$ $W$'s for each neuron as prior for the fully connected layer, we only have a single kernel for the convolution (we're not considering multiple channels yet). 

The $W$ essentially forms a hypothesis about the spatial representation of $X$ through a reduced number of parameters (shared weights). 

### Locality

The principle of locality is essentially saying that the important spatial features of a given image can be represented by a kernel, $W$, where the weights that correspond to the most important features in $X$ are central in the kernel $W$.

Thereby, for a given $W$, we can weight it's parameters outside a given range of $W_{ab}$, say $(W_{-\Delta, \Delta}, W_{-\Delta, \Delta})$ to $0$ (more of a conceptual understanding of locality that we have a kernel of a smaller size to consider locality of features... we don't actually pad $W$. We pad $X$)

This further reduces the number of parameters, enabling us to construct a robust hypothesis, if implemented right, of the spatial features of our image with a increased efficiency.

This introduces more shift invariance (as the important learned weighte become centrally concentraved within your kernel, $W$), where the learnt parameters will be able to detect the feature in an image despite being in a different location, but of course, the model will **not** be able to denote **where** in the feature map a given image is located, given that a smaller set of weights, the *kernel*, will only be able to learn local representations, it's degree of locality depending on the size of the Kernel.

This is great when we have a dataset that is built upon features that are translation invariant, meaning that despite moving around sets of features in the given image, they still belong to the same class label.

Therefore, the ConvNet, can learn from these translation invariant samples.

Of course, if the image isn't translation invariant, the principle of locality will fail as the model won't consider spatial locality and hence will be prone to misclassifying samples.

**TLDR**

- We can reduce the # of params for a model, via locality (padding) and shared weights
- With the decreased number of params, given the principle of locality, we're forming a hypothesis of weights, $W$, that are shift invariant.
- If we a have a shift invariant sample, i.e., an image that still belongs to the same class despite having objects within shifted around, then the hypothesis, $W$, will be able to generalize to those images.
- Inversely, if our images aren't shift invariant, then the hypthesis $W$, will fail to generalize given that it fails to enscaptulate the full spatial structure of the entire image at once.

### Channels!

In the case of images, convolutions must be defined over the 3 RGB channels that each image posseses (of course, unless MNIST but who does that anyway?).

So, rather than denoting $W$ as a simple $a\times b$ kernel, we must add a $3rd$ depth dimension.

Your input, $X$, in the first layer will have $3$ channels, and in the latter layers will $d$ different channels, depending on the $d$ amount of kernels we have for a given layer.

The convolution operation, considering a different number of channels then becomes (also considering principle of locality aka, padding the weights):


```math
Z_{i,j,d} = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} \sum_c W_{a,b,c,d}X_{i+a, j+b, c}
```

where 

- $a, b$ denote the shape of each kernel, $W$
- $c$ denotes the total # of channels in $X$ (input channels), which also corresponds to the total number of kernels (1 per channel) corresponding to the set of $dth$ set of kernels or $dth$ output. Each $c$ kernel, $W_c$, has a set of shared weights.
- $d$ denotes the total # output channels, also denoting the number of unique kernels, $W$, for each $X$. Each $d$ set of kernels, $W$, has a different set of weights.

> Note that the fact that each c kernel, W_c, has a set of shared weights is only true when we're considering regular convolutions for multiple channels. For ***Depthwise Convolutions***, we instead have a unique set of weights for each channel for each X input.

### Insights / Thoughts

- the principle of locality in convnets assumes shift invariance of your input samples. if they aren't shift invariant, you'll get a convnet that doesn't generalize.
- in depthwise convolutions we have separate filters with separate weights. we don't compute a weighted sum over all channels. if we have 3 input channels we have 3 output channels. 
  - while in a regular convolution, after each convolution (element wise mul + weighed sum) for each channel, we have a separate filters with unique weights, same as depthwise convs. but then once we compute the convolution for the I,j position, we sum the outputs for the 3 channels to get a single channel output for the 3 channels in $X$
- a single kernel in the earlier layers can learn to recognize edges in the input image thereby given that it recognizes edges, it can be reused across multiple positions of the input image to recognize other edges. combined with multiple kernels at later layers, we can construct from edges, more recognizable and important features as each later kernel is able to extract the important features that make up the image.