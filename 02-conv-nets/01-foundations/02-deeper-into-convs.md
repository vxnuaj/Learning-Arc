# Deeper into Convolutions

### Cross Corr. vs Convolution

Remember the prior defined convolution operation as:

```math
Z_{i,j,d} = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} \sum_c W_{a,b,c,d}X_{i+a, j+b, c}
```

where 

- $a, b$ denote the shape of each kernel, $w$
- $c$ denotes the total # of channels in $x$ (input channels), which also corresponds to the total number of kernels (1 per channel) corresponding to the set of $dth$ set of kernels or $dth$ output. each $c$ kernel, $w_c$, has a set of shared weights.
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

In this case rather than reversing the indexing of $X$, we isntead flip $W$ which turns to the equiv alent result. In mathematical proofs, $W$ is flipped such that writing proofs becomes simpler. In ML Libraries, we typically perform a cross correlation instead.

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
            Y[i, j] = (X[i:i+K_h, j:j+K_w] * K).sum() 
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

- more padding of the input -> larger output
- greater stride of the convolutional kernel -> smaller output

When performing convolutions, it's common that the edges of the input, $X$, are pixels that don't get used often. When you slide the Kernel, $K$, over the image, it covers a given edge pixel of $X$ only once.
Other pixels in $X$ are convolved over multiple times. 

> Won't lead to the kernel K properly learning the edges of the image. If the edges of the image might be important, you want to apply padding!

Another implication is convolving a larger $K$ over $X$ will ultimately lead to a larger reduction in pixel size of the feature map, as you get deeper into your network.

**Padding**, if needed can resolve the issue by adding 'pads' or zeros around the edges of $X$. 

To get the same size for your output, a size thatm atches your input, if the $k_h \times k_w$ dimensions of $K$ are are odd, we can denote padding $p$ as 

```math

p_w = \frac{k_w - 1}{2}
\\[3mm]
\text{and} 
\\[3mm]
p_h = \frac{k_h - 1}{2}

```

where $p$ is the amount of padding for one axis of $X$ ($h$ or $w$).

> if K has dims that are even, we won't be able to get the same height or width for the output unless we modify the stride to be something else than 1.

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

## Insights / Thoughts

- You could also train a convnet with pure corr operations.
- padding only certain sides?