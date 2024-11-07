## Identity Mappings in Deep Residual Networks

*"The difference in Resnet and ResNetV2 rests in the structure of their individual building blocks. In ResNetV2, the batch normalization and ReLU activation precede the convolution layers, as opposed to ResNetV1 where the batch normalization and ReLU activation are applied after the convolution layers."*

### Introduction

- We analyze deep residual netowrks by creating a "direct" path for propagating gradients -- not only within a given residual unit but through the entire network.

- Notation
  - $h(x_l)$ is the identity mapping, such that $h(x_l) = x_l$
  - $y_l$ is the output of the residual unit prior to $f$
  - $f$ is $\text{ReLU}$

- Derivations show that if $h(x_l)$ and $f(y_l)$ both become identity mappings, the gradient could be propagated from one unit to any other unit in forward and backward passes.
 
- After analyzing various verions of $h(x_l)$, such as $I$, gating (via activations), or $1 \times 1$, they found that the the skip connections as $I$ was the best for fastest error reduction and lowest training loss.

- Using Batchnorm $\text{BN}$ and $f$ as as pre-activation, prior to a convolution rather than after as an activation, allows for $f(y_l)$ and $h(x_l)$ to be identity mappings such that they lead to a new residual unit design.
  - They fine improvements on Cifar with a 1001-layer ResNet and ImageNet with a 200-layer ResNet

### Analysis of Deep Residual Networks

The original residual unit is given as:

```math

y_l = h(x_l) + \mathcal{F}(x_l, \mathcal{W_l})
\\[3mm]
x_{l + 1} = f(y_l)
```

If $f$ was an identity mapping, then we can rewrite as:

```math

x_{l+1} = x_l + \mathcal{F}(x_l, \mathcal{W_l})

```

and then recursively as:

```math

x_{l+2} = x_{l+1} + \mathcal{F}(x_{l+1}, \mathcal{W_{l+1}}) = x_l + \mathcal{F}(x_{l}, \mathcal{W_{l}}) + \mathcal{F}(x_{l+1}, \mathcal{W_{l+1}}) 

```

given that

```math

x_{l+1} = x_l + \mathcal{F}(x_l, \mathcal{W_l})

```

so it can be ultimately expressed as

```math

x_L = x_l + \sum_{i = l}^{L = 1} \mathcal{F}(x_i, \mathcal{W}_i) \tag{4}

```

- This can be done for any deeper unit, $l$.

> Essentially, given the original residual connection as $x_l$, you can simply concatenate the residual transformation, $\mathcal{F}(x_l, \mathcal{W}_l)$ for all $l$ layers up to the current $l$, such that you get back the current output for the current $l$ ... why did they do all that just to explain this lol.

> From now, $\mathcal{R_i}$ will denote the $ith$ residual unit, while $\mathcal{F_i}$ denonotes the actual function that represents the computation for the $ith$ residual unit.

- Equation $4$ has the property that any given output to a given $\mathcal{R}_i$ can be computed as the summation of all $\mathcal{R_{i - n}}$ where $0 < n < i$, and $x_0$ (original input to the start of the residual layers)
  
- Equation $4$ also has the property, where for any given layer, $l$ which contains a residual connection, the gradient with respect to a given input, $x$, will always be a at least the size of the gradient with respect to a given input $x$, for a later layer.
  - $(\frac{∂L}{∂x_5})(\frac{∂x_5}{∂x_3}) = (\frac{∂L}{∂x_5}) \cdot 1 = (\frac{∂L}{∂x_5})$
  - See [resgrad.md](resgrad.md) for more in depth.

### Importance of Identity Skip Connections

Identity Skip Connections become increasingly important for a residual network, rather than using parameterized skip connections via $1 \times 1$ convolutions or gated residual connections.
 
Consider:

```math

x_{l+1} = \lambda_l x_l + \mathcal{F(x_l, W_l)}

```

where $\lambda_l$ is any scalar that changes the magnitude of the residual connection.

Going with the recursive expression of the residual connections up the layer $L$:

```math

x_L = (\prod_{i = 1}^{L - 1} \lambda_i) x_l + (\prod_{i = 1}^{L - 1} \lambda_i) \sum_{j = {i+1}}^{L - 1}\mathcal{F}(x_i, W_i)
\\[3mm]
\text{or}
\\[3mm]
x_L = (\prod_{i = 1}^{L - 1} \lambda_i) x_l + (\prod_{i = 1}^{L - 1} \lambda_i) \hat{\mathcal{F}}(x_i, W_i)
```

where $\hat{\mathcal{F}}$ absorbs $\lambda$'s, backpropagation is defined as:

```math
\frac{∂L}{∂x_l} = \frac{∂L}{∂x_L}(\prod_{i=l}^{L}\lambda_i + \sum_{i = l}^{L - 1}\mathcal{F}(x_i, W))
```

Note that the scalar, $1$, has turnt into a scalar $\prod_{i=l}^{L}\lambda_i$, which can be arbitrarily big or small. For a smaller value, the backpropagated gradient, $\frac{∂L}{∂x_L}$, will be scaled into a smaller value such that vanishing gradients can become an issue. 

For a larger $\lambda$, the backpropagated gradient can become exponentially larger the more layers we backpropagate through such that exploding gradients diminish the quality of the model.

> $I$ = Identity Transformation

If $\lambda$ remains to be $1$, as is in the $I$ transformation, the gradient will remain unscaled as it is backpropagated through the network, such that $I$ residual connections become extremely important.

> Note that ResNet-110 doesn't consider $f = I$ as described above but simply $\text{ReLU}$.

- They compared training runs of the original $\text{ResNet-110}$ with a network that had a $\lambda = .5$.
- The training error was substantially higher for the models with $\lambda = .5$, indicating that transformations that $≠ I$ for a residual connection can hurt the gradient flow.

- Highway Network Gating Function: $y = T(x) \cdot H(x) + (1 - T(x)) \cdot x$

  - $T(x)$ is the learnt probability or amount of the output from the current layer we want to keep.

  - $x$ is the original input, to be multiplied by $1 - T(x)$, discarding the learnt % of $x$ for a given residual connection.

  - $T(x)$ is a learnt probability, denoted by the transformation $\sigma (W_T x + b_T)$ or for convolutions, $\sigma (\mathcal{K} * x + b_T)$, where $\mathcal{K}$ is size $1 \times 1$

- They compared the results of training a Highway Network work the original $\text{ResNet-110}$ and see that it's performance still lags behind, given the parameterized gating mechanism.
- They also test with a modified gating mechanism, where only the residual connection is gated as $(1 - T(x)) \cdot x$, still noting a suboptimal result.
-  They also test $1 \times 1$ as parameterized residual connections, showing poor results yet again.
-  Dropout also sucks.
- A model should be more expressive given more parameters yet $I$ still triumphs for residual connections. This shows that the issue doesn't lie in model expressiveness but rather optimization issues (shattering gradients).

### On the Usage of Activation Functions

- Prior, they conducted their experimnets with $f = \text{ReLU}$, not the Identity
- **BN + ReLU after Addition**: results are worse
- Now they move $f$ to be $I$
- **ReLU Before Addition**: not a good idea, given that we're learning residuals but $ReLU$ is monotonically increasing when $x > 0$, we only compute $+$ valued residuals such that the model isn't able to express the residuals when they are $-$ valued, which can limit the model.

**ReLU pre-activation / full pre-activation**:

Using a ReLU as the preactivation prior to the summation of the residual, such that $f$ turns into $I$ turns the residual equation as:

```math
y = I(x_l + F(x))
```

where where $I = f$ and $F$ is the transformation applied onto the input, which includes BatchNorm and ReLU.

This transformation, $F$ is: ReLU -> Conv -> BN -> ReLU -> Conv -> BN

Including BatchNorm prior, alongisde ReLU yields:


ReLU -> Conv -> BN -> ReLU -> Conv -> BN

such that our $\text{ReLU}$ activations enjoy the benefit of normalized inputs, so that the activated values don't become as large. This helps to reduce the co-variate shift and smooth the loss surface for better training (also aiding with exploding gradients).

<br>
<div align = 'center'>
<img src = 'https://miro.medium.com/v2/resize:fit:1400/0*r29KJfYQUoPKvDqG' width = 800>
</div>