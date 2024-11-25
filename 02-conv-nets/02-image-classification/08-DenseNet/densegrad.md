## Gradients for DenseNet

The forward pass for a DenseBlock can be given as:

```math

H_{\ell} = f_{\ell}([H_0, H_1, ..., H_{\ell - 1}]) \tag{1}

```

where the argument passed into the current, $\ell th$ layer, is a concatenation of all previous outputs with the current input within the given DenseBlock.

> Note that $\ell$ must be $0 < \ell < L$ ($L$ is total num of layers in the DenseBlock)

To compute the gradient for any given $H_{\ell}$, you first can note that a given $H_{\ell}$, as given in the function above, depend on all $H_{\ell - n}$ (where $0 < n < \ell$).

Also, $H_{\ell}$ contributes to all $H_{i + m}$ (where $0 < m < (L - \ell)$).

```math
H_{i + m} = f_{\ell}([H_0, ... , H_{\ell}, ..., H_{\ell + m - 1}]) \tag{2}
```

Say we have a loss, $\mathcal{L}$, and we want to take the gradient of the loss w.r.t to $H_{\ell}$.

```math

\frac{∂\mathcal{L}}{∂H_{\ell}}

```

> *This will be denoted as $∂H_{\ell}$ for simplicity*

To compute $∂H_{\ell}$, we need to account for it's gradient $\forall \hspace{1mm} \ell \in L$, in which $H_{\ell}$ contributed to an output.

> So using equation $(2)$, $\forall \hspace{1mm} m$

Then the total gradient (accumulated across all $\ell \in L$, where $H_{\ell}$ had a contribution) is simply a summation of all $∂H_{\ell}$, $\forall \ell \in L$, where $H_{\ell}$ had a contribution.

```math

\frac{∂\mathcal{L}}{∂H_{\ell}} = \sum_{j = \ell + i}^L (\frac{∂\mathcal{L}}{[∂H_{0}, ∂H_{1}, \dots, ∂H_{\ell}, \dots,∂H_{j-1}]})(\frac{[∂H_{0}, ∂H_{1}, \dots, ∂H_{\ell}, \dots,∂H_{j-1}]}{∂H_{\ell}}) \tag{3}

```

> $\ell$ is equal to the current layer. 
> 
> We do $j = \ell + 1$ in the $\sum$ as $H_{\ell}$ only had a contribution in layers after $\ell$, and not $\ell$.
> 
> We do $j - 1$ in the denominator of the first factor as if we did $H_L$, it wouldn't make mathematical sense as for the last layer of the DenseBlock ($L$), it's output can't contribute to itself.

Taking a look at the overall gradent flow, for $\ell = 3$ in a 5-layer DenseBlock:

```math

\frac{∂\mathcal{L}}{∂H_{\ell}} = (\frac{∂\mathcal{L}}{∂H_{\ell + 2}})(\frac{∂H_{\ell+2}}{∂H_{\ell + 1}})(\frac{∂H_{\ell+1}}{∂H_{\ell}})

```

it's easy to see how the vanishing gradient can be diminished, as we're accumulating gradients for the respective $∂H_{\ell+2}, ∂H_{\ell+1}, ∂H_{\ell}$, through equation $3$, meaning $\forall \hspace{1mm} \ell \in L$ that any given $∂H_{\ell + i}$ had a contribution such that the summation for the given $H_{\ell}$, leads to gradients of higher magnitudes.

Then when backpropagating using the chain rule, we're able to mitigate a vanishing gradient.

Albeit, given that we're accumulating gradients, a common conception might be the inverse, exploding gradients.

BatchNormalization exists for mitigating this very issue, such that inputs to the $\text{ReLU}$, don't become so large that $∂$'s begin to explode as we backpropagate deeper, into earlier layers.