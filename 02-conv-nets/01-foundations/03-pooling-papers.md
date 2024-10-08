## Stochastic Pooling

> *[Zeiler & Fergus](https://arxiv.org/pdf/1301.3557) - 2013*

**Abstract**

- Introduce as imple method for regularizing large convnets
- Repalce the deterministic pooling operations with a stochastic pooling procedure
- You randomly pick up the pooled feature according to a multinomial distribution, given by the activations in the receptive field.
- It is hyperparameter free and can be combined with dropout / data augmentation to further regularize a COnvnet
- SOTA on four image datasets, relative to other approaches that don't use data augmentation

**Introduction:**

- Neural Networks are prone to overfitting
- You can use $L_p$ regularization, weight decay, dropout, etc
- Dropout doesn't work well on ConvNets, likely because you're already reducing the numebr of weights via a Kernel size smaller than the input feature size.
- To incerase regualrization, you can instead use *stochastic pooling* which uses a multinomial distribution based on the receptive field of the pooling kernel, to stochastically pool a given activation, rather tha deterministically pooling the maximum value.

**Stochastic Pooling**

To perform stochastic pooling, first normalize the activations to compute the probabilities $p$ for each region $k \in R_j$, where $R_j$ is the $jth$ region:

```math

p_i = \frac{a_i}{\sum_{k \in R_j} a_k}

```

Then sample from the distribution based on probability $p$ to pick the pooled activation @ location $l$, given as:

```math

a_l, \text{where l} \sim P(p_1,...,p_{|R_j|})

```

**Stochastic Pooling during Test Time**

Rather than max-pooling as above during test time, you can isntead comptue a weighted averaging of the receptive field of the pooling kernel, with the weights based on the probability $p$ of the receptive field, which is computed exactly as above.

```math

a_l = \sum_{i \in R_j} p_ia_i

```

In essence can be seen as a means to add noise to the model, making sure the weights are able to account for and generalize better, despite the noise.

## Fractional Max-Pooling

> *[Benjamin Graham](https://arxiv.org/pdf/1412.6071) - 2014*

**Abstract**

- Max-pooling is typically done with an $\alpha \times \alpha$ kernel where $\alpha = 2$

- Constantly switching between Conv layers and max-pooling layers can be limiting in performance as there is a rapid loss in spatial size, especially when $\text{stride} = \alpha$

- Enter *fractional max-pooling*, where $\alpha$ can be a non-int... fraction! 

- A form of *stochastic* pooling (:^0)

**1.**

- Max Pooling $2 \times 2$ is very commonly used and popular, with $\text{stride} = 2$
- Disjointed nature (non overlapping max pooling operations), leads to a loss of information, which then leads to lack of generalization.

> Perhaps by having too little information, we lack to ability to create meaningful Kernels that are able to extract features (underfitting)

- Proposed solutions are $3 \times 3$ pooling layers with a stride of 2.
- Stochastic Pooling (see above)
- But the means to do so still reduce the dimensionality of the feature maps by a large factor. If pooling coudl reduce by $\sqrt{2}$, you could easily use twice as many pooling layers, makign it easier for kernels to learn and extract more valuable subsequent features after pooling
- The idea is to reduce the spatial size by a factor of $\alpha$ where $1 < \alpha < 2$

```math
\alpha = \text{pooling fraction, ratio between spatial size of input and output of pooling layer}
```

**Fractional Max-Pooling**

You can define each pooling region as:

```math

P = [a_{i-1}, a_i - 1] \times [b_{j-1}, b_j - 1]
\\[3mm]
\text{or}
\\[3mm]
P = [a_{i-1}, a_i] \times [b_{j-1}, b_j]

```

where $a_{i-1}, a_i$, denotes the range of the height dimension of the pooling kernel while $b_{j-1}, b_j$ denotes the range of the width of the pooling kernel

The given $a$ and $b$ are pseudorandom as: $a_i = ceiling(\alpha(i + u))$ where $\alpha \in (1, 2)$ and $u \in (0, 1)$

> this is so weird.