# Residual Networks

Say $\mathcal{F}$ is the set of functions that our neural network can precisely reach.

For all $f \in \mathcal{F}$, there is a set of parameters, $\Theta$ that can be obtained by training the neural network.

Assume $f^\text{*}$ is the truth function we need to properly classify all $x \in \mathcal{X}$ to the proper $y \in \mathcal{Y}$.

If $f^\text{*}$ resides within $\mathcal{F}$, then the model can reach $f^\text{*}$ and it's appropriate for the problem we're solving with the neural network.

Typically, we aren't so lucky so we attempt to find a given $f_{\mathcal{F}}^\text{*}$, which represents the best approximation for $f^\text{*}$, subject to $f_\mathcal{F}^\text{*} \in \mathcal{F}$.

If we're able to switch to a more powerful architectural design, say $\mathcal{F}'$, we should be able to arrive at a beter outcome for approximating $f^\text{*}$, as $f_{\mathcal{F}'}^\text{*}$.

But say, $\mathcal{F}' \not\in \mathcal{F}$, meaning it is not a subset of $\mathcal{F}$.

There is no guarantee that $\mathcal{F}'$ won't deviate in a random direction away from the optima.

> This is the issue with non-nested function classes, instead you should be expanding on your current architecture (treating it as your best explanation for the proper function space), to get a better $\mathcal{F'} \in \mathcal{F}$

While if $\mathcal{F}' \in \mathcal{F}$, we'll only be searching within the existing function space of $\mathcal{F}$, such that we can land within a more precise approximation of a better $f^\text{*}$, as $f^\text{*}_{\mathcal{F}'}$, ultimately improving on our existing hypothesis.

> This assumes that $\mathcal{F}$ is already our best hypothesis for the function space that contains the best $f^\text{*}_{\mathcal{F}}$, such that we can iterate by choosing a $\mathcal{F'} \in \mathcal{F}$. Otherwise, we wouldn't neccesarily want $\mathcal{F}' \in \mathcal{F}$ and might want to search within an entirely different function space.

Say we construct $\mathcal{F}'$ such that it is larger than $\mathcal{F}$ and $\mathcal{F} \in \mathcal{F}'$. This strictly increases the expressiveness of our neural network, while still containint the original best explanation of the function space $\mathcal{F}$. 

Then, we may be able to get a better approximation for $f^\text{*}$ without destroying our original hypothesis.

> You want to make your models deeper, building on your best explanations for the bset architecutre. This is why initial conditions become your root cause for building the best solutions.

### ResNet

Say you have a transformation, $f: x \rightarrow y$, where $x$ is a given low-resolution input image and $y$ is the desired higher-resolution output image.

(the goal is to increase the resolution of the input).

Rather than learning the parameters, $w$, that increase the resolution of $x$ such that we get $y$, we could very easily learn the difference between $y$ and $x$, through parameters $w$ of $f(x)$

Then, $y = x + f(x)$ as $f(x)$ is simply the difference between $y$ and $x$.

This is essentially a **residual connection**, where we add the original input to the output, such that $f(x)$ is essentially learning the **residuals** of the input from the desired output and then adding onto $x$ such that we get the desired $y$ by filling the gap.

This holds the assumption that $x$ already holds the neccesary information to reach to $y$, such that we don't need a drastic transformation, say $\mathbb{R}^m \rightarrow \mathbb{R}^n$, thereby you only have a small **residual**.

> Note that for a residual block, the spatial dimensions $H \times W$ must remain the same for the residual connection.

In very deep neural networks, as a ConvNet gets deeper, it'll typically forget (discard) of the information in the input for hierarchically extracted **relevant** features that contribute to maximizing for the correct output, $y$, such that the neural network forgets what the original $x$ (or pays no regard) is constructed as.

Adding a residual connection allows for the model, at every layer, to reference the previous input as a model for learning it's parameters, $w$, with more context from the original input, to only model the residuals.

<div align = center>
<img src = resblock.png>
</div>

A given residual block can learn the identity function, when the model finds that it is optimal to model $y = x$. Thereby, the kernels, $\mathcal{K}$ for a given convnet will be sparse, full of $0$s if $y = x$

Yet this doesn't cause the vanishing gradient problem due to the nature of the residual connection as $y = f(x) + x$. 

During backpropagation, a residual connection helps reduce the effect of vanishing gradients as given the skip connection, we're adding an extra constant to the gradient of $\frac{∂F(x)}{∂x}$ such that when we backpropagate via the chain rule, our gradients don't become as small.

Given the residual connection as $y = F(x) + x$, the derivative of the loss $\mathcal{L}$ with respect to $x$ can be computed as:

```math

\frac{∂L}{∂x} = (\frac{∂L}{∂y})(\frac{∂y}{∂x})
```

where $\frac{∂L}{∂y}$ is the gradient w.r.t to the output activation.

Then

```math

\frac{∂y}{∂x} = \frac{∂F(x)}{∂x} + \frac{∂x}{∂x} = \frac{∂F(x)}{∂x} + 1

```
as a derivative of $x$ with respect to itself is purely $1$.

This added constant to the gradient allows for a greater gradient to be propagated backward of at minimum a magnitude of $1$ such that we're able to mitigate and perhaps even eliminate a vanishing gradient, even when we have a sparse $\mathcal{K}$ full of $0$s.

In theory the residual connetions allow for easier optimization, as given that we're capable of learning $f$ as $I$, then we won't have an overly paramterized $\mathcal{K}$, such that the loss landscape becomes smoother, diminishing the problems that come with local minima, saddle points, and longer computational times.

<br>
<div align = 'center'>
<img width = 500 src = 'https://miro.medium.com/v2/resize:fit:723/1*_Qd_txKxRlsMdfuH2J-k4g.png'/>
</div>
<br>

We aren't forced to learn complex transformations denoted by a complex $\mathcal{K}$, that we're able to learn only the **minimum differences** between a given input and output, then making it easier to approximate functions with only small differences and mitigating an issue with overfitting.

**Other Benefits**

- Reusing Features at later layers from earlier layers.
- Facilitating Deeper Networks by mitigating Vanishing Gradients and "remembering" earlier learnt features.

## Deep Residual Learning for Image Recognition

### Abstract

- Deep Neural Networks are hard to train

    > Implying Vanishing Gradients

- In ResNet, layers are reformulated as residual functions with reference to the input layers.

    > Residual Functions refer to a functoin that represents the difference between the observation and the prediction.

### Introduction

- As models get deeper, $∂$'s $\rightarrow 0$, such that the deeper the model is, the smaller a given $∂$ will be at earlier layers, due to the nature of the chain rule. 
  - Thereby, given the update rule $\Theta = \Theta - \alpha * ∂$, for a deeper network it is possible that we end up with larger training error due to the inability to converge onto an optima as the $∂$'s are too small for meaningful updates.

<div align = 'center'>  
<img width = 500 src = err.png>
</div>

- Residual Connections **beat** deeper models, deeper models didn't have comparable training error to models with residual connections.
- Given the target, $\mathcal{H}(x)$, the model learns the residual, $\mathcal{F}(x):= \mathcal{H}(x) - x$ and then constructs an output as $\hat{\mathcal{H}}(x) = \mathcal{F}(x) + x$.

### Deep Residual Learning

- Residual Connections are simply $y = f(x) + x$, where $f(x)$ represents a multi-layer neural network (feed forward or convolutional).
- The vanilla Residual Connection works when the input is same size as the output
  - If the connection connects an $x$ that is differemt dimension to to $f(x)$, you can pad $f(x)$.
  - Or you can create a weighted matrix, $W_s$ such that $y = f(x) + W_sx$ such that $W_s$ performs the linear transformation of $x$ into $\mathbb{R}^n$, where $n$ is the dimensionality of $f(x)$.
    - Expressed via convolutions, this can be equivalent to $1 \times 1$ convolutions with $\mathcal{K}$, where we decrease the channel size to the desired channel count.
    - Then $W_s$ or $\mathcal{K}$ can be learnt parameters.
    - In the ResNet shown in the paper, their $1 \times 1$ convolutions were applied with a stride of $2$ such that they effectively reduce the dimensions of the model to properly fit the desired size.
    - They use $\text{PCA}$ pixel based RGB augmentation (see [here](https://www.vxnuaj.com/blog/PCAaug))
    - They use Batchnorm after each convolution prior to each activation and after eacih convolution.
    - They use Xavier Initialization.

## Identity Mappings in Deep Residual Networks

*"The difference in Resnet and ResNetV2 rests in the structure of their individual building blocks. In ResNetV2, the batch normalization and ReLU activation precede the convolution layers, as opposed to ResNetV1 where the batch normalization and ReLU activation are applied after the convolution layers."*

### Introduction

- We analyze deep residual netowrks by creating a "direct" path for propagating gradients -- not only within a given residual unit but through the entire network.

- Notation
  - $h(x_l)$ is the identity mapping, such that $= x_l$
  - $y_l$ is the output of the residual unit prior to $f$
  - $f$ is $\text{ReLU}$

- Derivations show that if $h(x_l)$ and $f(y_l)$ both become identity mappings, the gradient could be propagated from one unit to any other unit in forward and backward passes.
 
- After analyzing various verions of $h(x_l)$, such as $I$, gating (via activastion), or $1 \times 1$, they found that the the skip connections as $I$ was the best for fastest error reduction and lowest training loss.

- Using Batchnorm $\text{BN}$ and $f$ as as pre-activation, prior to a convolution rather than after, allows for $f(y_l)$ and $h(x_l)$ to be identity mappings such that they lead to a new residual unit design.
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

x_L = x_l + \sum_{i = l}^{L = 1} \mathcal{F}(x_i, \mathcal{W}_i)

```

- This can be done for any deeper unit, $l$.

> Essentially, given the original residual connection as $x_l$, you can simply concatenate the residual transformation, $\mathcal{F}(x_l, \mathcal{W}_l)$ for all $l$ layers up to the current $l$, such that you get back the current output for the current $l$ ... why did they do all that just to explain this lol.

