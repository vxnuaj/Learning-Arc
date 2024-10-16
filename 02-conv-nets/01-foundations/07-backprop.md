
### Backprop for CNNs

> [Resource](https://people.tamu.edu/~sji/classes/bp.pdf)

Backprop for CNNs is similar to the backpropagation for fully connected feed forward neural networks, as they both use the multivariable chain rule, but given that a conv layer shares weights, we can instead compute the gradient for individual $w_i$ parameter in the kernel, $\mathcal{K}$, as the summed gradient of the given $w_i$ across all possible positions of $\mathcal{K}$, during the convolution of $\mathcal{K} * Z^l$, where $Z$ is the input to the $l\text{th}$ layer.

Let's assume the $lth$ layer is a convolutional layer for now.

Prior to doing so, as is done in a fully connected model, you'd need $∂Z^l$ which in the case of a convnet, is simply a transposed convolution between the kernel at the $lth$ layer, $\mathcal{K}^l$, and the gradient of the loss w.r.t $Z$ at the $l+1$ layer, $∂Z^{(l+1)}$.

The transposed convolution (sometimes known as the deconvolution purely for the following reason) puts back the elements of an output in a convolution, back to it's original spatial position prior to the convolution.

For example:

```math
W * X = Z
```

is the convolution of kernel, $W$, slid over $X$ to yield $Z$.

The transposed convolution computes:

```math

W \hspace{1mm}\hat{*}\hspace{1mm}Z = \hat{X}

```

where $\hat{X}$ is the same shape as $X$. The values of the transposed convolution, don't yield the same values as the original $X$. Just the same spatial dimensions, expanding values of $Z$ back onto the same original positions $\hat{X}$.

So, going back, $∂Z^{l}$ is simply the backpropated gradient w.r.t to the weighted sum at $l$, attained via the transpose convolution.

So where in a fully connected layer you have:

```math

∂Z^l = (\frac{∂L}{∂A^{l+1}})(\frac{∂A^{l+1}}{∂Z^{l+1}})(\frac{∂Z^{l+1}}{∂A^l})(\frac{∂A^l}{∂Z^l}) = (W^{(l+1)})^T ∂Z^{(l+1)} \odot ∂\text{act-func(}Z^{l})

```

through a convolutional layer, we have:

```math

∂Z^{l} = (\frac{∂L}{∂A^{l+1}})(\frac{∂A^{l+1}}{∂Z^{l+1}})(\frac{∂Z^{l+1}}{∂A^l})(\frac{∂A^l}{∂Z^l})=  \text{conv}^{\text{Transpose}}(\mathcal{(K^{l+1})}^{\text{Flipped}}, ∂Z^{(l + 1)}) \hspace{.5mm} \odot ∂\text{act-func}(Z^{l})

```

where

- $\text{conv}(\mathcal{(K^{l+1})}^{flipped}, ∂Z^{(l + 1)})$ is the same dimensions as $Z^l$.
- $∂\text{act-func}(Z^{l})$ is the derivative of the activation w.r.t to $Z^l$, $\frac{∂A}{∂Z^l}$

We flip the kernel, $\mathcal{K}$, as that allows the gradients to be correctly propagated back into the right positions to update the weights, $w$.

Ultimately, we have $∂Z^l$, the gradients w.r.t to the output weighted sum of the convolutional layer.

Once we have $∂Z^l$, we can compute $∂W^l$.

To get $∂W^l$, given $∂Z^l$, the kernel $\mathcal{K}^l$, and the input to the current layer, $A^{l-1}$, we essentially slide the kernel $\mathcal{K}^l$ over $A^{l-1}$ to extract the $ith$ $A_{patch}^{l-1}$, as is done in the forward pass for the convolutional layer, but rather than computing a weighted sum, we simply element wise multiply with the $ith$ element in $∂Z^l$ at each $i$ location to get a portion of our gradient at each $i$th step, say $\mathcal{O}_i$

Once we have every possible $\mathcal{O}_i$, we element wise sum all $\mathcal{O}_i$, to get our gradient for $\mathcal{K}^l$, as $∂W^l$.

```math

\mathcal{O}_{ij} = A^{l-1}_{patch_{ij}} \odot ∂Z_{ij}^l\\[3mm]
∂W^l = \sum_n^N\mathcal{O}_{n}

```

where $N$ is the total number of patches.

Ultimately, we can express this as:

```math

∂W^l = \sum_n^N(A^{l-1}_{patch_{ij}} \odot ∂Z_{ij}^l, \forall i, j)

```

where $i$ is the index which denotes is the $i$th patch in $A$ (same dimensions as kernel, $\mathcal{K^l}$) or the $i$th element in $Z$.

We can also compute $∂B^l$ from $∂Z^l$ as:

```math

∂B_{ij}^l = \sum_{i = 0}^I \sum_{j = 0}^{J} ∂z_{ij}^l

```

Thereby, if we have a batch size that is greater than $1$, we can average all our gradients over the batchsize as:

```math
∂\Theta =\frac{1}{B} \sum_{s=0}^B ∂\Theta_s
```

where $B$ is the total number of samples in the batch.

> *Note that, while we're summing over spatial positions, $\text{M (row)}$ and $\text{N (column)}$, we don't average over them. Each gradient has a unique value for each given parameter, $b_{ij}$ and $w_{ij}$, averaging would kill the unique gradient for each parameter, making each weight update equivalent.*

### Backpropagation through Max-Pooling

Gradients for **max pooling** layers can be a tricky headache at times, especially in scenarios where you have a max pooling layer right before a fully connected layer, and you want to get the gradients for both the max-pooling layer and the convolutional layer prior to it.

In regards to getting the gradients, $∂z^l$, assuming the current layer, $l$, is the max pooling layer and we're propagating back from a fully connected layer, we can simply compute it as:

```math

∂Z^l = (W^{l+1})^T∂Z^{l+1}

```

or if propagating back from a convolutional layer,


```math

∂Z^{l} = \text{conv}^{\text{Transpose}}(\mathcal{(K^{l+1})}^{\text{Flipped}}, ∂Z^{(l + 1)})


```

without the need for the hadamard product with $∂\text{act-func}(Z^l)$ that would've been needed if we were computing a $∂Z$ within a fully connected layer. This is as the output of the max pooling layer, $Z^l$, makes no use of such an activation function and therefore, we don't need to compute the gradient of the activation during backprop **because we don't have one**.

Now that we have $∂Z^{l}$, the gradient w.r.t the output of the max pooling layer, things get a little more tricky, when we want to get the $∂$'s w.r.t to the inputs of the pooling layer, or equivalently, w.r.t to the outputs of the prior convolutional layer, $l - 1$.

Given that **max pooling** downsamples the input to a smaller output ($\mathbb{R}^{m\times n} \rightarrow \mathbb{R}^{\hat{m} \times \hat{n}}$), we need to find a way to upsample or "undo" the max pooling operation within backpropagation. We want to upsample. Typically, in a conv layer, you'd do so via a transpose convolution with kernel $\mathcal{K}$, but max pooling doesn't make use of a weighted kernel $\mathcal{K}$, rendering it unviable.

First off, to upsample, we need to remember the indices of the maximum values within the pooling region of a given input $Z$, that were yielded via the kernel in the max pooling layer, $\mathcal{K}^{\text{MP}}$.

Then, when we perform the max unpooling, we create an empty mask ($∂Z^{l-1}_{mask}$) with same shape as of the input of a pooling layer $Z$. Then, with the cached indices, we identify the respective positions in this empty mask and insert each value from the derivative of the max pooling layer, $∂Z^l$, into it's corresponding positions in $∂Z_{mask}^{l-1}$. Then $∂Z_{mask}^{l-1} \rightarrow ∂Z^{l-1}$.

This is essentially, **max-unpooling**, with with our partial derivatives.

Then afterward, to follow the chain rule of calculus, as would've been done in the gradient in a fully connected layer, $∂Z^l = (W^{l+1})^T∂Z^{l+1} \odot ∂\text{act-func}(z^{l})$ we compute the hadamard product with $∂\text{act-func}(Z^{l-1})$.

```math

∂Z^{l-1} = \text{max-unpool}(Z, \text{idxs}, ∂Z^{l}) \hspace{1mm}\odot \hspace{1mm} ∂\text{act-func}(Z^{l-1})

```

The indivdual gradient values of $∂Z^{l-1}$ are just the values of $∂Z^{l}$ itself, as in the forward pass, when we compute $\text{max}(x \in X_{n:\mathcal{K_n}, m:\mathcal{K_m}}), \forall m, n$ we're essentially returning the maximum value of the given region of $X$ covered by the kernel, $\mathcal{K}$, with **no weights** applied onto the input. This can be seen as being equivalent to having weights $= 1$, such that the original formulation for what would've been the gradient as $∂Z^{l-1} = W^l \cdot \text{max-unpool}(Z, \text{idxs}, ∂Z^l)$ becomes $∂Z^{l-1} = 1 \cdot \text{max-unpool}(Z, \text{idxs}, ∂Z^l)$.

### Backpropagation through Average-Pooling

Backprop through average-pooling is similar, yet simpler than backpropagating gradients through max-pooling layers.

Just as prior, to get gradients, $∂z^l$, assuming the current layer $l$ is the average pooling layer, we can compute as:

```math

∂Z^l = (W^{l+1})^T∂Z^{l+1}

```

or if propagating back from a convolutional layer,


```math

∂Z^{l} = \text{conv}^{\text{Transpose}}(\mathcal{(K^{l+1})}^{\text{Flipped}}, ∂Z^{(l + 1)})

```

Now to propagate the gradient, $∂Z^l$, back to a prior layer, say a convolutional layer, we have to **upsample**, just as was done when propagating gradients back through the max pooling layer.

Though fortunately, this is simpler than max-unpooling, as we don't have to store the indices of the max-pooled values with respect to the original input in the cache.

Instead, given the kernel for the average-pooling layer, $\mathcal{K}^l$, and the gradients, $∂Z^l$, we can instead create an empty mask of same dimensions as $Z^{l-1}$ (just as was done in $\partial$ through max-pooling), let's call it $∂Z^{l-1}_{mask}$. We can slide $\mathcal{K}^l$ over $∂Z^{l-1}_{mask}$, and for each $r$ region in the mask, $∂Z^{l-1}_{mask_{r}}$, we extract the $r$th value in $∂Z^l$ at position $i, j$, and evenly spread it out over the region.

If the $∂Z^{l-1}_{mask_r}$ was dimensions $2 \times 2$, or size $4$, and the $r$th value in $∂Z^l$ was $8$, we'd divide $8$ by $4$, yielding $2$, and insert the $2$ into all $4$ slots of the current region.

If our stride is less than the a given dimension of the kernel, (whether it be height or width), at some point, we will a have overlapping regions to insert values of $∂Z^l$. Then the case becomes that for the overlapping portions, we add the newly dispersed gradients, $(\frac{∂Z^l}{r_{size}})^r$, to the previously dispersed gradients at the previous $r$, $(\frac{∂Z^l}{∂r_{size}})^{r-1}$.

This is done for all possible regions, $r$, in $∂Z^{l-1}_{mask}$, then $∂Z^{l-1}_{mask} \rightarrow ∂Z^{l-1}$

```math

∂Z^{l-1} = ∂Z^{l-1}_{mask_r} + \frac{∂Z^{l-1}_{i, j}}{r_{size}}, \forall\hspace{1mm} (r, (i, j))
\\[3mm]
\text{(this is element wise sum, where }\frac{∂Z^{l-1}_{i, j}}{r_{size}} \text{ is broadcasted over the region)}
```










