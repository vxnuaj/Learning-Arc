## Upsampling

Learn:

- [X] Nearest Neighbor Interpolation
- [X] Max Unpooling
- [X] Bilinear Interpolation
- [X] Bicubic Interpolation

**Upsampling**, in the context of Convolutional Neural Networks, is the opposite of down sampling (pooling), where instead of reducing the dimensionality of our activations, we want to increase the dimensionality of the input feature maps.

A specific application for this is image segmentation, such as semantic, instance, or panoptic segmentation, which makes use of an encoder-decoder architecture.

In the encoder architecture, you extract the high level features of the input image and encode it into a lower dimensional feature map, $\mathbb{R}^n \rightarrow \mathbb{R}^d$.

The decoder, then applies a set of convolutions, just like the encoder, and **upsamples** the encoded image, rather than downsampling, ultimately decoding the original image into a segmented image. 

If we want to get segmented feature maps of the same dimensionality as the input, $\mathbb{R}^d \rightarrow \mathbb{R}^n$, we **must** upsample the encoded features.

This can be done via

- max-unpooling
- bilinear interpolation
- bicubic interpolation
- nearest neighbor interpolation

The reason we downsample in the first place is such that we need to extract the most important features (*CNNs are featue extractors!*), from a larger receptive field (**encoding the image**).

If we never downsampled, we would never be able to extract the features (output of encoder) from a larger receptive field (original image).

### Nearest Neighbor Interpolation

```math

X = \begin{bmatrix}1, 2 \\ 3, 4\end{bmatrix}\\[3mm]

T:X \rightarrow \mathbb{R}^{16} \text{ (or } \hat{X})\\[3mm]

\hat{X} = \begin{bmatrix} 1, 1, 2, 2 \\ 1, 1, 2, 2 \\ 3, 3, 4, 4 \\ 3, 3, 4, 4\end{bmatrix}


```

When you might have competing nearest neighbros such as:

```math

X = \begin{bmatrix}1, 2 \\ 3, 4\end{bmatrix}\\[3mm]

T:X \rightarrow \mathbb{R}^{9} \text{ (or } \hat{X})\\[3mm]

\hat{X} = \begin{bmatrix} 1, X, 2 \\ X, X, X \\ 3, X, 4\end{bmatrix}


```

you can determine the interpolated values in the feature map, $X$, via any rule, as long as it is consistent. Of you course, you couldn't average, it has to be based on a neareset neighbor for it to be considered nearest neighbor interpolation.

$X$ could be the left nearest neighbor as a rule for instance. Or a right nearest neighbor.

### Bi-Linear Interpolation

```math

X = \begin{bmatrix} 1, 2, 3 \\ 4, 5, 6 \\ 7, 8, 9\end{bmatrix}\\[3mm]

T: X \rightarrow \mathbb{R}^{25}\\[3mm]

X = \begin{bmatrix} 1.0, 1.5, 2.0, 2.5, 3.0 \\ 2.5, 3.0, 3.5, 4.0, 4.5\\ 4.0, 4.5, 5.0, 5.5, 6.0 \\ 5.5, 6.0, 6.5, 7.0, 7.5\\  7.0, 7.5, 8.0, 8.5, 9.0\end{bmatrix}\\[3mm]

```

### Bed of Nails Unpooling

```math

\text{At layer l}
\\[3mm]
X = \begin{bmatrix} 1, 2, 3 \\ 4, 5, 6 \\ 7, 8, 9\end{bmatrix}\\[3mm]
\\[3mm]
K = \begin{bmatrix}K, K \\ K, K\end{bmatrix}, \text{ stride = 1}
\\[3mm]
\text{Pool(X, K)}
\\[3mm]
\hat{X} = \begin{bmatrix}5, 6 \\ 8, 9\end{bmatrix}
\\[3mm]
\text{Decoder, unpooling at layer L - l:}
\\[3mm]
X = \begin{bmatrix}5, 6 \\ 8, 9\end{bmatrix}
\\[3mm]
\text{Max-Unpool(X)}
\\[3mm]
\hat{X} = \begin{bmatrix} 5, 6, 0 \\ 8, 9, 0 \\ 0, 0, 0 \end{bmatrix}

```

Puts the pooled values into the grid section of the original kernel (in this case, $2 \times 2$), while filling the uneeded postiions with $0$s.

### Max-Unpooling

Smarter version of bed of nails unpooling, remembering the original positions of the max-pooled values.

```math

\text{At layer l}
\\[3mm]
X = \begin{bmatrix} 1, 2, 3 \\ 4, 5, 6 \\ 7, 8, 9\end{bmatrix}\\[3mm]
\\[3mm]
K = \begin{bmatrix}K, K \\ K, K\end{bmatrix}, \text{ stride = 1}
\\[3mm]
\text{Pool(X, K)}
\\[3mm]
\hat{X} = \begin{bmatrix}5, 6 \\ 8, 9\end{bmatrix}
\\[3mm]
\text{Decoder, unpooling at layer L - l:}
\\[3mm]
X = \begin{bmatrix}5, 6 \\ 8, 9\end{bmatrix}
\\[3mm]
\text{Max-Unpool(X)}
\\[3mm]
\hat{X} = \begin{bmatrix} 0, 0, 0 \\ 0, 5, 6 \\ 0, 8, 9 \end{bmatrix}

```

Max-unpooling helps in preserving the spatial structure that was lost in pooling, now in the decoder architecture, to reconstruct the prior image, for segmentation.  

We don't need to unpool the original features, so we can easily introduce sparsity by adding $0$s. This is as the most important features are denoted by the extracted features via the MaxPooling. Thereby, we only need to reocnstruct with the most important features to properly segment the image.
