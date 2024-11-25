# Gradients for Residual Connections

> *I don't understand things if I can't write them out from first principles*

Here's how we always have a $∂$ of at least $\frac{∂L}{∂x_{l+2}}$ for the $lth$ layer with a Residual Connection at every other $l$, and why we need the Identity Transformatoin to maintain this important feature of residual networks, for a very **deep network**.

> Feel free to use this as a resource to understand, *["Identity Mappings in Deep Residual Networks"](https://arxiv.org/pdf/1603.05027)*.


<div align = center>

### Forward

```math
x_2 = \text{ReLU}(F_1(x_1)) 
```
```math
x_3 = \text{ReLU}(F_2(x_2) + x_1)  \tag{1}
```
```math
x_4 = \text{ReLU}(F_3(x_3))
\\[3mm]
```
```math
x_5 = \text{ReLU}(F_4(x_4) + x_3) \tag{2}
```
```math
L = \text{CrsEnt}(p \cdot log(x_5))
```

### Backward

```math

\frac{∂L}{∂x_5} = x_5 - p
\\[5mm]
\frac{∂L}{∂x_4} = (\frac{∂L}{∂x_5})(\frac{∂x_5}{∂x_4})
```
```math
\frac{∂L}{∂x_3} = (\frac{∂L}{∂x_5})(\frac{∂x_5}{∂x_4})(\frac{∂x_4}{∂x_3}) + (\frac{∂L}{∂x_5})(\frac{∂x_5}{∂x_3}) = (\frac{∂L}{∂x_5})(\frac{∂x_5}{∂x_4})(\frac{∂x_4}{∂x_3}) + (\frac{∂L}{∂x_5})(\frac{∂F(x_4)}{∂x_3} + 1) \tag{∂2}
```
```math
\frac{∂L}{∂x_2} = (\frac{∂L}{∂x_5})(\frac{∂x_5}{∂x_4})(\frac{∂x_4}{∂x_3})(\frac{∂x_3}{∂x_2})
```
```math

\text{(EQ. ∂1 BELOW)}\\[3mm]
\frac{∂L}{∂x_1} = (\frac{∂L}{∂x_5})(\frac{∂x_5}{∂x_4})(\frac{∂x_4}{∂x_3})(\frac{∂x_3}{∂x_2})(\frac{∂x_2}{∂x_1}) + (\frac{∂L}{∂x_5})(\frac{∂x_5}{∂x_4})(\frac{∂x_4}{∂x_3})(\frac{∂x_3}{∂x_1}) = (\frac{∂L}{∂x_5})(\frac{∂x_5}{∂x_4})(\frac{∂x_4}{∂x_3})(\frac{∂x_3}{∂x_2})(\frac{∂x_2}{∂x_1}) + (\frac{∂L}{∂x_3})(\frac{∂F(x_2)}{∂x_1} + 1)
```

Notice:

```math
(\frac{∂L}{∂x_5})(\frac{∂x_5}{∂x_3}) = (\frac{∂L}{∂x_5})(\frac{∂F(x_4)}{∂x_3} + 1)

\\[3mm]
\text{and}
\\[3mm]
(\frac{∂L}{∂x_5})(\frac{∂x_5}{∂x_4})(\frac{∂x_4}{∂x_3})(\frac{∂x_3}{∂x_1}) = (\frac{∂L}{∂x_3})(\frac{∂F(x_2)}{∂x_1} + 1)
```

as when we take the $∂$ w.r.t to $∂x_1$ or $∂x_3$ propagating to the residual connection, we end up taking the gradient of $∂x_1$ and $∂x_3$ w.r.t to itself, such that it is equal to $1$, then the gradient becomes $(\frac{∂F(x_{l+1})}{∂x_l} + 1)$

When we multiply with the other gradient that forms the chain rule, $\frac{∂L}{∂x_5}$, $\frac{∂L}{∂x_3}$, or more generally, $\frac{∂L}{∂x_{l+2}}$, we're able to propagate back a $∂$ of at least $≥ \frac{∂L}{∂x_{l+2}}$

Simpler example:

```math

\frac{∂y}{∂x_l} = \frac{∂}{∂x_l}(F(x_{l+1}) + x_l) = \frac{∂F(x_{l+1})}{∂x_l} + \frac{∂x_l}{∂x_l} = \frac{∂F(x_{l+1})}{∂x_l} + 1

```

This exposes the inner mechanics behind computing gradients for $x_3$ and $x_1$ -- where above we had $\frac{∂y}{∂x_l}$, in context of the former example, at equation $(∂1)$, it's equivalent to $\frac{∂x_3}{∂x_1}$, where $∂x_3 = ∂y$ if we compare to the simpler example. Correspondingly, for equation $(∂2)$, $\frac{∂y}{∂x_l} = \frac{∂x_5}{∂x_3}$.

Hence, for residual connections, $F(x_l) + x_{i - 1}$, the gradient with respect to an $x_l$ will always be $\frac{∂F(x_{l+1})}{∂x_l} + 1$, that is if we have the $I$ transformation for a residual connection, and then the overall gradient w.r.t to $x_l$ will always be $≥ \frac{∂L}{∂x_{l+2}}$. (Worst case scenario, it being equivalent to $\frac{∂L}{∂x_{l+2}}$ if $\frac{∂F(x_{l+1})}{∂x_l} = 0$)

Thereby, for the given layers that include a residual $I$ connection, in this case those which involve $F_4$ and $F_2$, we'll always have a $∂ ≥ \frac{∂L}{∂x_{l+2}}$, such that we end up mitigating the vanishing gradient problem for those layers, and perhaps earlier layers, though this is dependent on how large your residual block is.

You can view this as a gradient "highway" (reminiscent of highway networks)

The deeper your residual block is, the more layers the $∂$'s will backpropagate through without the alleviating residual connection, such that your set of gradients can still vanish, until the gradients meet a residual connection.

Now consider:

```math

x_{l+1} = \lambda_l x_l + F(x_l)

```

where $\lambda_l$ is any scalar that changes the magnitude of the residual connection.

Note that a Residual Connection can be recursively defined for any layer $L$ as:

```math
x_{L} = x_l + \sum_{i = l}^L \mathcal{F}(x_{i}, \mathcal{W_{i}})
```
 
Introducing $\lambda$ and going with the recursive expression of the residual connections up the layer $L$:

```math
x_L = (\prod_{i = 1}^{L - 1} \lambda_i) x_l + \hat{\mathcal{F}}(x_i, W_i)
```

where $\hat{\mathcal{F}}$ absorbs $\lambda$'s, backpropagation is defined as:

```math
\frac{∂L}{∂x_l} = \frac{∂L}{∂x_L}((\prod_{i=l}^{L}\lambda_i) + \frac{∂}{∂x_l}\sum_{i = l}^{L - 1}\mathcal{F}(x_i, W))
```

Note that the scalar, $1$, has turnt into a scalar $\prod_{i=l}^{L}\lambda_i$, which can be arbitrarily big or small. 

> From now: <br><br>
> $\prod_{i = l}^L \lambda_i = \lambda\\[3mm]$
> $I$ = Identity Transformation

For a smaller $\lambda$, the backpropagated gradient, $\frac{∂L}{∂x_L}$, will be scaled into a smaller value such that vanishing gradients can become an issue. 

We lose the property of: $∂ ≥ \frac{∂L}{∂x_L}$

For a larger $\lambda$, the backpropagated gradient can become exponentially larger the more layers we backpropagate through such that exploding gradients diminish the quality of the model.

If $\lambda$ remains to be $1$, as is in the $I$ transformation, the gradient will remain unscaled as it is backpropagated through the network, such that $I$ residual connections become extremely important.

This is why having residual connections be transformation via $1 \times 1$ convolutions or some other transformation is damaging to backpropagation. You reduce the expressiveness of a **deep** network.

> Some Thoughts.