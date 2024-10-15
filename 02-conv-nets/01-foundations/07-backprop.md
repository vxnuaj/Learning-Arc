
### Backprop for CNNs

> [Resource](https://people.tamu.edu/~sji/classes/bp.pdf)

Backprop for CNNs is similar to the backpropagation for fully connected feed forward neural networks, as they both use the multivariable chain rule, but given that a conv layer shares weights, we can instead compute the gradient for individual $w_i$ parameter in the kernel, $\mathcal{K}$, as the averaged gradient of the given $w_i$ across all possible positions of $\mathcal{K}$, during the convolution of $\mathcal{K} * Z^l$, where $Z$ is the input to the $l\text{th}$ layer.

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

where $\hat{X}$ is the same shape as $X$. The values of the transposed convolution, don't yield the same a the original $X$. Just the same spatial dimensions, expanding values of $Z$ back onto the same original positions $\hat{X}$.

So, going back, $∂Z^{l}$ is simply the backpropated gradient w.r.t to the weighted sum at $l$.

So where in a fully connected layer you have:

```math

∂Z^l =  (W^{(l+1)})^T ∂Z^{(l+1)} \odot ∂\text{act-func(}Z^{l})

```

in a convolutional layer, we have:

```math

∂Z^{l} = \text{conv}^{\text{Transpose}}(\mathcal{(K^{l+1})}^{\text{Flipped}}, ∂Z^{(l + 1)}) \hspace{.5mm} \odot ∂\text{act-func}(Z^{l})

```

where

- $\text{conv}(\mathcal{(K^{l+1})}^{flipped}, ∂Z^{(l + 1)})$ is the same dimensions as $Z^l$.
- $∂\text{act-func}(Z^{l})$ is the derivative of the activation w.r.t to $Z^l$, $\frac{∂A}{∂Z^l}$

We flip the kernel, $\mathcal{K}$, as that allows the gradients to be correctly propagated back into the right positions to update the weights, $w$.

Once we have $∂Z^l$, we can compute $∂W^l$.

To get $∂W^l$, given $∂Z^l$, the kernel $\mathcal{K}^l$, and the input to the current layer, $A^{l-1}$, we essentially slide the kernel $\mathcal{K}^l$ over $A^{l-1}$ to extract the $ith$ $A_{patch}^{l-1}$, as is done in the forward pass for the convolutional layer, but rather than computing a weighted sum, we simply element wise multiply with the $ith$ element in $∂Z^l$ at each $i$ location to get a portion of our gradient at each $i$th step, say $\mathcal{O}_i$

Once we have every possible $\mathcal{O}_i$, we element wise sum all $\mathcal{O}_i$, to get our gradient for $\mathcal{K}^l$, as $∂W^l$.

```math

\mathcal{O}_i = A^{l-1}_{patch} \odot ∂Z_i^l\\[3mm]
∂W^l = \sum_i^I\mathcal{O}_i
```

Ultimately, we can express this as:

```math

∂W^l = \sum_i(A^{l-1}_{patch_i} \odot ∂Z_i^l)

```

where $i$ is the index which denotes is the $i$th patch in $A$, same dimensions as kernel, $\mathcal{K^l}$, or the $i$th element in $Z$.

We can also compute $∂B^l$ from $∂Z^l$ as:

```math

∂B_{ij}^l = \sum_{i = 0}^M \sum_{j = 0}^{N} ∂z_{ij}^l

```

Thereby, if we have a batch size that is greater than $1$, we can average them over the batchsize as:

```math
∂\Theta =\frac{1}{B} \sum_{s=0}^B ∂\Theta_s
```

where $B$ is the total number of samples in the batch.

> *Note that, while we're summing over spatial positions, $\text{M (row)}$ and $\text{N (column)}$, we don't average over them. Each gradient has a unique value for each given parameter, $b_{ij}$ and $w_{ij}$, averaging would kill the unique gradient for each parameter, making each weight update equivalent.*

### Backpropagation through Max-Pooling

Gradients for **max pooling** layers can be a tricky headache at times, especially in scenarios where you have a max pooling layer right before a fully connected layer, and you want to get the gradients for both the max-pooling layer and the convolutional layer prior to it.

In regards to getting the gradients, $∂z^l$, assuming the current layer, $l$, is the max pooling layer, we can simply compute it as:

```math

∂Z^l = (W^{l+1})^T∂Z^{l+1}

```

without the need for the $∂\text{act-func}(z^l)$ that would've been needed if we were computing a $∂z$ within a fully connected layer. This is as the output of the max pooling layer, $z^l$, makes no use of such an activation function and therefore, we don't need to compute the gradient of the activation during backprop **because we don't have one**.

Now that we have $∂z^{l}$, the gradient w.r.t the output of the max pooling layer, things get a little more tricky, when we want to get the $∂$'s w.r.t to the inputs of the pooling layer, or equivalently, w.r.t to the outputs of the prior convolutional layer, $l - 1$.

Given that **max pooling** downsamples the input to a smaller output ($\mathbb{R}^{m\times n} \rightarrow \mathbb{R}^{\hat{m} \times \hat{n}}$), we need to find a way to upsample or "undo" the max pooling operation within backpropagation. We want to upsample. Typically, in a conv layer, you'd do so via a transpose convolution with kernel $\mathcal{K}$, but max pooling doesn't make use of a weighted kernel $\mathcal{K}$, rendering it unviable.

First off, to upsample, we need to remember the indices of the maximum values within the pooling region of a given input $Z$, that were yielded via the kernel in the max pooling layer, $\mathcal{K}^{\text{MP}}$.

Then, when we perform the max unpooling, we create an empty mask ($∂Z^{l-1}_{mask}$) with same shape as of the input of a pooling layer $Z$. Then, with the cached indices, we identify the respective positions in this empty mask and insert each value from the derivative of the max pooling layer, $∂Z^l$, into it's corresponding positions in $∂Z_{mask}^{l-1}$. Then $∂Z_{mask}^{l-1} \rightarrow ∂Z^{l-1}$.

This is essentially, **max-unpooling**, with with our partial derivatives.

Then afterward, to follow the chain rule of calculus, as would've been done in the gradient in a fully connected layer, $∂Z^l = (W^{l+1})^T∂Z^{l+1} \odot ∂\text{act-func}(z^{l})$ we compute the hadamard product with $∂\text{act-func}(Z^{l-1})$.

```math

∂Z^{l-1} = \text{max-unpool}(Z, \text{idxs}, ∂Z^{l}) \hspace{1mm}\odot \hspace{1mm} ∂\text{act-func}(Z^{l-1})

```

The indivdual gradient values of $∂Z^{l-1}$ are just the values of $∂Z^{l}$ itself, as in the forward pass, when we compute $\text{max}(x \in X_{n:\mathcal{K_n}, m:\mathcal{K_m}}), \forall m, n$ we're essentially returning the maximum value of the given region of $X$ covered by the kernel, $\mathcal{K}$, with **no weights** applied onto the input. This can be seen as being equivalent to having weights $= 1$, such that the original formulation for what would've been the gradient as $∂Z^{l-1} = W^l \cdot \text{max-unpool}(Z, \text{idxs}, ∂Z^l)$ becomes $∂Z^{l-1} = 1 \cdot \text{max-unpool}(Z, \text{idxs}, ∂Z^l)$.