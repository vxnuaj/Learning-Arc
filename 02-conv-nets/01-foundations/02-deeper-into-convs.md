# Deeper into Convolutions

### Cross Corr. vs Convolution

Remember the prior defined convolution operation as:

```math
Z_{i,j,d} = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} \sum_c W_{a,b,c,d}X_{i+a, j+b, c}
```

where 

- $a, b$ denote the shape of each kernel, $w$
- $c$ denotes the total # of channels in $x$ (input channels), which also corresponds to the total number of kernels (1 per channel) corresponding to the set of $dth$ set of kernels or $dth$ output. each $c$ kernel, $w_c$, has a set of different weights.
- $d$ denotes the total # output channels, also denoting the number of unique kernels, $w$, for each $x$. each $d$ set of kernels, $w$, has a different set of weights.
- $[-\Delta, \Delta]$ denotes the range of 'padding' we're applying to our input kernel (typically not done in practice)

This is actually the cross-correlation operation, typically used in signal processing.

A typical convolution operation is defined as:

```math
Z_{i,j,d} = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} \sum_c W_{a,b,c,d}X_{i-a, j-b, c}
```   

The difference between both being is the order of which we apply the the convolution operation onto $X$. Prior we were indexing $X$ as an addition of $i + a$ and $j+ b$. Essentially, from the center of $X$ as $X_{00}$, we're moving to the bottom-right position first. In the formal definition of the convolution in context of deep learning, the convolution takes the $X$ from the top left to bottom right, starting with $X_{i-a, j-b}$.

There isn't so much of a difference between both operations, just that we're indexing $X$ in a different manner. Ultimately, both will yield an equivalent result, given the same set of weights $W$, with perhaps just the positions of $Z$, say $i, j$ will now be in positions $-i, -j$

Or we can perform:

```math

Z_{i,j,d} = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} \sum_{c} W_{-a, -b, c, d} X_{i + a, j + b, c}

```

In this case rather than reversing the indexing of $X$, we isntead flip $W$ which turns to the equiv alent result. In mathematical proofs, $W$ is flipped such that writing proofs becomes simpler, as the operation becomes commutative. In ML Libraries, we typically perform a cross correlation instead.

*Many machine learning libraries implement cross-correlation but call it convolution.In this text we follow this convention of calling both operations convolution andspecify whether we mean to ﬂip the kernel or not in contexts where kernel ﬂippingis relevant. In the context of machine learning, the learning algorithm will learnthe appropriate values of the kernel in the appropriate place, so an algorithm basedon convolution with kernel ﬂipping will learn a kernel that is ﬂipped relative to thekernel learned by an algorithm without the ﬂipping. - deep learning book*

> [cool](https://www.youtube.com/watch?v=C3EEy8adxvc)

The output of a cross correlation is just the flipped version of the output of the convolution. The weights are flipped for the two.

### Output Dims.

For an input $X$ and a kernel $W$ we can define the output shape ($Z$) as:

```math

Z_H = \frac{X_H + 2P - V_H}{S} + 1\\[3mm]
Z_W = \frac{X_W + 2P - V_W}{S} + 1\\[3mm]
\text{Output Dims: } Z_H \times Z_W

```

where $\text{Matrix}_H$ denotes the height of the given matrix and $\text{Matrix}_W$ is the width of the given matrix.

$V$ is the kernel.

$S$ is the stride, $P$ is the padding, and the $+1$ denotes the initial position of the Kernel.

Say I have a $4 \times 4$ $X$ and a $2 \times 2$ $V$ with no stride or padding.

```math

3 = \frac{4 + 2(0) - 2}{0} + 1\\[3mm]
3 = \frac{4 + 2(0) - 2}{0} + 1\\[3mm]
\text{Output Dims: }3 \times 3
```

Now say I have the same matrices but stride of $2$

```math

2 = \frac{4 + 2(0) - 2}{2} + 1\\[3mm]
2 = \frac{4 + 2(0) - 2}{2} + 1\\[3mm]
\text{Output Dims: }2 \times 2

```

Here is a convolution:

```code
def conv2d(X, K): # no padding / stride
    X_h, X_w = X.shape
    K_h, K_w = K.shape
    X = torch.flip(X, dims = (0, 1)) # assuming weights aren't flipped
    Y = torch.zeros(size = (X_h - K_h + 1, X_w - K_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+K_h, j:j+K_w] * K).sum() self.X[i * stride:i * stride + kernel_size].
    return Y
```

### Edge Detector!

Given $X = \begin{bmatrix}0, 1, 2, 2 \\ 0, 0, 1, 2 \\ 0, 0, 0, 1\end{bmatrix}$ and $K = \begin{bmatrix}1, -1\end{bmatrix}$, and apply a convolution, we get:

```math
\begin{bmatrix}- 1, -1, 0 \\ 0, -1, -1 \\ 0, 0, -1\end{bmatrix} 
```

We're essentially learning edges of $X$ with $K$

### Padding and Stride

Given an $n \times n$ matrix, $X$, and an $k_h \times k_w$ kernel, $K$, the larger $K$ is the smaller the output feature map, $Z$ will be. This is as the kernel is able to apply a weighted ***sum*** to a broader area of spatial features in the input $X$.

You can control the output size via 

- larger kernel -> smaller output
- more padding of the input -> larger output
- greater stride of the convolutional kernel -> smaller output

When performing convolutions, it's common that the edges of the input, $X$, are pixels that don't get used often. When you slide the Kernel, $K$, over the image, it covers a given edge pixel of $X$ only once.
Other pixels in $X$ are convolved over multiple times. 

> Won't lead to the kernel K properly learning the edges of the image. If the edges of the image might be important, you want to apply padding!

Another implication is convolving a larger $K$ over $X$ will ultimately lead to a larger reduction in dimensionality of the feature map, as you get deeper into your network.

**Padding**, if needed can resolve the issue by adding 'pads' or zeros around the edges of $X$. 

To get the same size for your output, a size thatm atches your input, if the $k_h \times k_w$ dimensions of $K$ are are odd, we can denote padding $p$ as 

```math

p_w = \frac{k_w - 1}{2}
\\[3mm]
\text{and} 
\\[3mm]
p_h = \frac{k_h - 1}{2}

```

where $p$ is the amount of padding for one axis of $X$ ($h$ or $w$), on one side (right or left | up or down).

> if K has dims that are even, we won't be able to get the same height or width for the output unless we modify the stride to be something else than 1.

Padding, of a minimum of $p_w$ and $p_h$ allows for all the edge features of the image to be considered / weighted as often as the central features. Otherwise, we'd perform a convolution over edges once and multiple times over the central pixels, depending on the size of your kernel.

Typically, CNN architectures have odd kernels, to make outputting the same size simple and easy.

For a convolutional kernel that has odd dims and a 2-dim input tensor, $X$, if we pad as:

```math

p_w = \frac{k_w - 1}{2}
\\[3mm]
\text{and} 
\\[3mm]
p_h = \frac{k_h - 1}{2}

```

the output, $Y$ at position $i,j$, can be easily shown to be the cross correlation / convolution of the kernel $K$ with $X$ where the center of $K$ is at $X_{ij}$

This can make convolutions eaiser to interpret and make sure that $Y$ captures the $ij$ feature of $X$ while maintaining the spatial hierarchy / resolution of the original input.

**Stride** reduces the size of the output feature map, with the benefit that it reduces computational cost, by reducing the number of convolutions and gradient calculations to train a model. This might allow for a model to be less prone to overfitting and generalize better, though it depends on the context of the architecture.

Also helps accelerate training given less computation time.

### Multiple I/O Channels

For RGB Images (and perhaps other types of data), you'll have multiple input and output channels. RGB are made of **RED, GREEN,** and **BLUE** channels, making an input size of $3 \times m \times n$, where $3$ is the number of channels, $m$ is the height of your image, and $n$ is the pixel width of your image. 

If you input the RGB image of $3$ channels, the kernel, $K$ needs to have $3$ channels to convovle over the 3 color channels of the input, $X$.

$K$ is then shape $c \times m \times n$, where $c$ is the channel dimension (depth) of the tensor. When we have a given $c_i$, the kernel then is simply a 2-dimensional tensor (matrix), convoluted onto the $c_i$ channel of $X$.

Given an input with 3 channels as:

```math
X = \begin{bmatrix}\begin{bmatrix}1, 2 \\ 3, 4\end{bmatrix} \begin{bmatrix}5, 6 \\ 7, 8\end{bmatrix}\begin{bmatrix}9, 10 \\ 11, 12\end{bmatrix}\end{bmatrix}
```

we have a $K_{cmn}$ as:

```math
K = \begin{bmatrix}\begin{bmatrix}-1, -1 \\ 1, 1\end{bmatrix} \begin{bmatrix}-2, 2 \\ -2, 2\end{bmatrix}\begin{bmatrix}-3, 3 \\ -3, 3\end{bmatrix}\end{bmatrix}
```

then we apply the convolution of a given $K_{c_{i}mn}$ as:

```math
Z = \sum_{i = 1}^C X_{c_imn} * K_{c_imn}

```
where $C$ is the total number of channels in $K$ and/or $X$

The final output $Z$ is simply the convolution of all corresponding channels of Kernels and Inputs, summed together element wise.

> this is not the case the depthwise convs, where we keep each channel separate.

In the case of increasing the total number of output channels, denote it as $C$, we can introduce multiple sets of kernels for a single RGB input of $3$ channels.

Say we had the RGB image $X$, of size $6 \times 6$, and a correpsonding kernel of size $2 \times 2$, with the size of our output channels to be $3$.

Then our tensor, $K$, will be a shape of $3 \times 3 \times 2 \times 2 \hspace{2mm} (\text{Kernel Set, Channel Count, Height, Width})$ 

Typically as we get into deeper layers of a ConvNet, we increase the # of channels while downsampling the resolution of each channel.

What this does from beginning to end is, that it allows each subsequent set of Kernels to extract more meaningful features from a given input.

Say you have 3 channels in your input image (RGB) and 2 sets of 3 kernels.

```math
X\hspace{2mm}(\text{1, 3, m, n}) \hspace{2mm}|\hspace{2mm}K \hspace{2mm}\text{(2, 3)}, k_m, k_n
```

Rather than viewing each channel of $X$ as a separate channel that represents different features of the image, you should see the different channels of $X$ ultimately working jointly together, alongside $K$, to ultimately extract the next important set of features in the weighted sum (convolution operation):

```math

\text{NOTATION NOT THE EXACTLY THE SAME AS ABOVE}
\\[3mm]
Z_{i,j,d} = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} \sum_{c} K_{-a, -b, c, d} X_{i + a, j + b, c}

```

Then in the weighted sum, the $d$ total output channels can denote a separate feature extracted from the original $X$, which is then typically broken down into multiple distinct feature maps through the addition of sets of Kernels, each Kernel extracting anotehr set of features through the distinct feature maps.

**TLDR**

- Multiple Channels in an input $X^l$ ($l$ is the layer) work together jointly alongisde a given kernel, $K$, to compute a weighted sum, resulting in a separate feature map. 
- For multiple sets of Kernels, $K_{i3k_mk_n}$, where $i$ is the $ith$ set of Kernels, each $ith$ set of Kernels works together jointly with the distinct channels to output the $dth$ feature map, which corresponds to an extracted feature.
- Each $K_{i3k_mk_n}$ learns to jointly work together with different channels to output a single feature, via the $dth$ feature map.
- Each channel of a given $dth$ $X^l$ can be seen as corresponding to a different extracted feature, where each set of $K$ learns to further extract more abstract features as we get deeper into the ConvNet.
- All channels combine via a weighted sum with a given $K$ to get the extracted feature. For different $K$, you will get different extracted features

### $1 \times 1$ Convolution??

$1 \times 1$ Convs are simply the weighted sum of all $3$ channels of a given input, $X^l$, with a given set of params in $K$.

Say we have an RGB image ($3$ channels), and a single set of kernels, $K$, of size $1 \times 1$ (obv. has 3 Kernels, $K_i$ in the set.).

When we convolve over the channels in the image, we're essentially combining the $ij$ pixel values in all 3 channels in the image through a combination with the corresponding weights in $K$ (here, there woudl be 3 params in $K$, one per channel given depth of 3 in the image).

```math
RGB_{1ij}K_1 + RGB_{1ij}K_2 + RGB_{1ij}K_3 = Z_{ij}
```

The output, $Z$ is then the same size as the input $RGB$, but has only 1 channel. We can have multiple channels in the output if we introduce multiple sets of $K$

$1 \times 1$ convolutions serve as a means to reduce the dimensionality of a given input, $X^l$, at the $lth$ layer, in order to setup the next, say for example, $5 \times 5$ convolution at the next layer, to use lower computational power and speed up training. 

While it might be useful to use $1 \times 1$ convs, it also reduces the complexity of your model as you don't have as many weights in a given $K_i \in K$ to represent the features of each channel in $X^l$. Instead, after the $1 \times 1$ convolution, we're concatenating all channels together via a linear combination / weighted sum, thereby we lose the number of unique parameters for each feature in the deep channels, as the feature in the deep channels no longer exist. All we're left with is a single feature and a single parameter for it after the $1 \times 1$ conv.

Aside from dimensionality reduction, $1 \times 1$ convolutions also serve as a means to compress the expressive information of the channels down into a single feature map, and if a non-linearity is applied, it can be used to learn an inter-channel relationship in the input. Applying multiple $1 \times 1$ convolutions can then be useful, as you learn different inter-channel relationships (or features), given a higher expressive power with more $K$, while still being able to drastically reduce dimensionality in a computationally efficient manner. Of course, they wont be as useful for having your ConvNet fit to data, but can still be an efficient tool when dealing with overfitting, low computational resources, or slow training.

**Computational Expense**

For an image of size $m \times n$, computing a $k \times k$ convolution is about:

```math

\mathcal{O}(m \times n \times k^2)

```

as we have a filter with area $k^2$, slid over the entire $m \times n$ area of the input feature map (assuming stride is $1$).

If we factor in multiple input ($c_i$) and output ($c_o$) channels, this turns into:

```math

\mathcal{O}(m \times n \times k^2 \times c_i \times c_o)

```

as we have to compute a convolution over $c_i$ filter maps with $c_o$ sets of kernels.

### Receptive Field

The receptive field of a given CNN is essentially the total amount of features the model sees to comptue a single feature at a given layer, not only from the direct previous layer, but from the initial layer. *How many features from the input has the convolution operations used to generate the given $ith$ feature in the $lth$ layer*.

For a given output feature, you can compute the receptive field with respect to any layer given the formula:

```math

r_i = r_{i-1} + (k - 1) \prod_{k=1}^{i - 1}s_k

```

where $r_i$ is the receptive field of teh $ith$ layer, $k$ is the kernel size, and $s_k$ is the stride of the $kth$ kernel.

To comute without iterating through each $r_i$, you can go as:

```math

r_l = 1 + \sum_{i=1}^l((k_i-1)\prod_{k=1}^{i-1}s_k)

```

Keep in mind that the receptive field corresponds to a specific kernel in a given layer, $k_i^l$ (forget previous notation).

For a given $k_i \in K_i^{lc}$ ($k_i$ is the specific kernel, $K_i^{lc}$ is the set of $k_i$ for the given output channel, $c$ at the $l$th layer.), larger receptive fields, given by a larger $k_i$ are able to summarize more information form a given feature map. If in the input layer, a larger receptive field will allow to us to handle a more global context from the input image, allowing us to denote the position of spatial features.

Say you have $10$ output channels, each with dims $2 \times 2$. The receptive field for each kernel is $25$. Then each kernel, $k_i$, is extracting a specific feature from that $25 \times 25$ area in the input layer, to then serve as a feature for the subsequent fully connected layers for softargmax classification (typically). 

Essentially, each input feature into the fc layers are extracted features from the original $25 \times 25$ receptive field. The more kernels you have, for a given $lth$ layer, the more features you can extract from the receptive field, $r_l$

>[!IMPORTANT]
>**A larger kernel: More global representations of the given feature map**\
>**More output channels (sets of kernels per channel): More features extracted from the given receptive field.**

### Pooling

As we get deeper into a ConvNet, after much dimensionality reducction (via typical convolutions or pooling), the smaller a given feature map becomes and the larger proportion the kernel covers relative to the given input feature map.

> (larger receptive field)

This is typical and ideal as in the final output layer, a convnet wants to take into account **all** features in the input to the hidden layer, in order to properly classify from the extracted features, which class $X$ belongs into.

Additionally, you also want the lower level features to be shift invariant, as in images, objects rarely appear in the same place as in other images. You want your convnet to be able to detect the same object no matter where it is located in the image.

This is **pooling**!

In pooling, you have a kernel with no parameters that slides over your feature map, $X^l$. Whislt sliding over your feature map, for the given receptive field, the kernel summarizes the information in a way that preserves information while reducing dimensionality.

*Average* pooling or *max* pooling.

**MAX POOLING**

For the given receptive field of $K^l$ on $X^l$, in *max pooling*, we aim to aggreate the info by taking the maximum value on the receptive field and outputting it onto the output feature map.

**AVERAGE POOLING**

For the given receptive field of $K^l$ on $X^l$, in *max pooling*, we aim to aggreate the info by taking the average value on the receptive field and outputting it onto the output feature map.

Pooling allows us to introduce a tad bit of shift invariance to a model. The degree of shift invariance can vary depending on the construction of the given feature map ... for example, if we have a $3 \times 3$ max-pooling layer vs a $2 \times 2$ max-pooling layer, the $3 \times 3$ max-pooling layer is always going to have a higher degree of shift invariance.

Though, it isn't immune to shift invariance, the probaiblity of it is higher. Of course, there could always be the chance that the maximum value within the receptive field of the pooling kernel on $X$ lies on the edge of the kernel. In that case, any shift to $X$ would not yield the pooling layer to be shift invariant.

Though, adding pooling layers **forces** a set of kernels to capture the features that are **shift invariant**. During backpropagation, if the Kernel hasn't captured the correct set of features that lead to the correct output, due to a lack of shift invariance in the set of captured features by the respective Kernel, the kernel will learn weights that allow for shift invariance to remain present as we go through the pooling layers. This is as we **need** a set of shift invariant features, to properly classify samples.


**Strides and Padding for Pooling**

- Higher stride to the pooling operation means a higher degree of reduced dimensionality
- More padding to the operation means a lesser degree of reduced dimensionality

**Pooling Multiple Channels**

- We pool multiple channels separately, no need to concatenate each channel as is done in convolutions.

### Convolutions and Pooling hold Very Strong Priors

Convolutions assume:

- Shift equivariance
- Locality via the Receptive field.
- Shared weights across hidden units can detect the same features

Pooling assumes:

- Shift invariance -- up to small shifts in the input feature map

Both form a hypothesis of the data you want to learn. The architecture matters more than a fully connected network, given that you have a smaller set of weights per feature and introduce pooling layers.

Per deeplearningbook.com:

*If a task relies on preserving precise spatial information, then using pooling on all features can increase the training error.*

You risk underfitting if your architecture isn't designed with the strong priors that reflect the data.


## Other

There's also **mirror padding** which essentially, for a given padding setting, $p$, reflects the pixels corresponding on the inner side of the edge onto the outer padded pixels.

Example:

```math

X = \begin{bmatrix}1, 2, 3 \\ 4, 5, 6 \\ 7, 8, 9\end{bmatrix}
\\[3mm]
\text{mirror padding with }p=1
\\[3mm]
X_{mirror} = \begin{bmatrix} 5, 4, 5, 6, 5 \\ 2, 1, 2, 3, 2 \\ 5, 4, 5, 6, 5 \\ 8, 7, 8, 9, 8 \\ 5, 4, 5, 6, 5 \end{bmatrix}

```

This can serve as a means for:

- Data Augmentation (eliminate edges of a sample and mirror pad)
- Preserving information of the edges of the image
- Smoother transitions at the borders

## Insights / Thoughts

- You could also train a convnet with pure corr operations.
- padding only certain sides?