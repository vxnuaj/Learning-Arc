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