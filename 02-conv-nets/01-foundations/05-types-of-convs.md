## Dilated Convolutions

Dilated Convolutions add a spacing, $d$ or a dilation rate, between different values in a given kernel, $K$.

Typically, in ConvNets, we want to increase the size of the receptive field, relative to the size of output feature maps for a given layer, $l$.

This is as we want to extract the ***relevant*** features to the input, in the output feature maps, $\hat{Y}^{output}$, such that we only include the ***most important*** features that are needed to classify the image in the subsequent $1 \times 1$ convolutions (or fully connected layers).

> This is why ConvLayers are known as feature extractors.

This can be done via:

- Pooling
- Strided Convs

although both reduce the resolution of the features, which can effectively reduce the expressive power of your ConvNet.

You can also increase the **kernel size** but that increases the # of parameters in your neural net, making it computationally expensive for large models.

You can instead use ***dilated convolutions***.

Where the dialation rate $= 1$, you have a standard convolution. For dilation rate, $d > 1$, there is a spacing of size $d - 1$ between each parameter in the kernel. There can be different dilation rates for the height and width of a given kernel, $d_h, d_w$ respectively.

Your kernel, then has a larger receptive field for the given $l$th layer, while still having the same parameter count, thereby not increasing the computational cost but actually decreasing it given a decreased amount of convolution operations.

Although for an individual convolution operation, at position $i, j$, you will skip over features in the feature map, in subsequent operations, say $i +1, j+1$, you will be convolving over the initially skipped features, thereby you don't lose as much information as you'd if you'd MaxPooled. 

While to increase size of receptive field relative to number of features in an output feature map, dilation can help towards doing so as the larger dilated kernel summarizes more info, given a larger receptive field, while losing less information (as strides can leave out pixels and max pooling leaves out features for the maximum), with lower computational cost than strided convolutions.

## Transposed Convolution

A transposed convolution (also known as a deconvolution or a fractionally strided convolution), is a convop that is commonly thought as the reverse of the standard convolution operation, as a 'deconvolution', but isn't truly the case.

While the transposed convolution can upsample a given $\hat{X}$ back into the original shape of $X$, where $X$ is the matrix prior to being convolved with kernel $\mathcal{K}$, it will not retain the same information as the original $X$, hence it isn't appropriate to call it a deconvolution.

> even if u use the same kernel, will be different **X**

For example, where

```math
W * X = Z
```

is the convolution of kernel, $W$, slid over $X$ to yield $Z$.

The transposed convolution computes:

```math

W \hspace{1mm}\hat{*}\hspace{1mm}Z = \hat{X}

```

Say you have $\hat{X}$ as the $X$ post-convolution with $\mathcal{K}$.

```math

\text{Shape } \hat{X}: (3 \times 3)\\[3mm]

\hat{X} = \begin{bmatrix}
x_{11} & x_{12} & x_{13} \\
x_{21} & x_{22} & x_{23} \\
x_{31} & x_{32} & x_{33}
\end{bmatrix}
\\[3mm]
\text{Shape } \mathcal{K}: (2 \times 2) \\[3mm]
\mathcal{K} = \begin{bmatrix}
k_{11} & k_{12} \\
k_{21} & k_{22}
\end{bmatrix}
```

The output size for the transposed convolution can be denoted as:

```math
O=S×(I−1)+K−2P
\\[2mm]
4 = 1 \times 2 + 2
\\[2mm]
Y \text{ shape: } (4 \times 4)

```

Lets initialize $Y$ to be all $0$s.

To perform the transposed convolution, you get a given $\hat{x}_{mn}$, where $m$ is the row index and $n$ is the column index, and element-wise multiply it with $\mathcal{K}$.

For the first op, you'd get:

```math

y_{m:\mathcal{K_m},\hspace{.2mm}n:\mathcal{K_n}} = \hat{x}_{11}\mathcal{K} = \hat{x}_{11} * \begin{bmatrix}k_{11}, k_{12} \\ k_{21}, k_{22}\end{bmatrix}

```

and the same is done for every $x_{mn}$.

Each $\hat{x}_{mn}$ in $\hat{X}$, corresponds to a portion of the output, $Y$, as $Y_{m:\mathcal{K_m},\hspace{.2mm}n:\mathcal{K_n}}$, where the portion of $Y$ is simply the same size as $\mathcal{K}$.

You can think of the output, $Y$, being a result of sliding the kernel over the given output matrix, $Y$, and adding all possible $Y_{m:\mathcal{K_m},\hspace{.2mm}n:\mathcal{K_n}}$ to the respective indices, $m:\mathcal{K_m},\hspace{.2mm}n:\mathcal{K_n}$ (think numpy slicing).

Overlapping slices will get some values of the transposed convolution operation added to values of the transposed convolution at previous indices, $m-1$ and $n-1$.



