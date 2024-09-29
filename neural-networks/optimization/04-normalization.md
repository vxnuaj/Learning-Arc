# Normalization

In deep learning, we typically train models by updating all layers during single iteration of the update rule, $\theta_{t+1} = \theta_t - \alpha g_t$.

Thereby, when we update a given layer, we're updating with the assumption that the other layer still has the parameters prior to the update. Of course, this isn't the case, we're updating all parameters for all layers during a given iteration.

When we work with very deep models, updating all layers are once can then lead to unpredictable results, if we aren't careful (i.e., set a smaller learning rate).

> *In practice, we update all of the layers simultaneously.*
> *When we make the update, unexpected results can happen because many functions composed together are changed simultaneously, using updates that were computed under the assumption that the other functions remain constant.* -- deep learning book.

The deeper your network gets, the greater change there will be in the update, as we're computing higher order partials for multiple layers, thereby increasing the unpredictability the output for a neural network $\hat{y}$ will yield after the weight update.

While on the first layer we can approximate gradients, via Taylor Series approximation, and perhaps on the second layer, to analytically compute the weight update and then the new value of $\hat{y}$, it becomes more difficult to consistently do so as our network gets deeper.

> Higher order optimization methods try to fix this, but even then aren't a fix to extremely deep models. They are also computationally expensive

Given the unpredictability of weight updates, there are certain weight updates that will be of disproportional magnitude to other weight updates in different layers. Some layers might have extremely large gradietns while others might not.

This is covariate shift, and can affect the stability and convergence of training.

### BatchNorm

BatchNorm provides a solution to this, as it bounds the range of the activations for each given layer, while still retaining their meaning in context of the problem the neural network is trying to optimize for.

Thereby, the gradients for a given neural network, aren't as large and the magnitude of the update for a given $\hat{y}$ becomes more predictable and less stochastic. It scales down the weight update, via a learnable distribution of activations for a given layer, $\sigma$ and $\mu$, where $\sigma$ is the variance of the activations and $\mu$ is the mean.

Given an activation matrix for the given $i$th layer, $A_i$, you can compute the normalized activations as the $z$-normalized matrix:

```math
A_i' = \frac{A_i - \mu_i}{\sigma_i}
```

$\mu$ and $\sigma$ are learned via gradient descent, exactly like any other parameter in the model is.

$\mu$ is constructed from $A_i$, shape $(\text{samples} \times \text{activations})$, by summing over the activations over all samples as:

$\mu = \frac{1}{m} \sum_i^M {A_i}$

While $\sigma$ is constructed in the same manner, in terms of axis operation, but computed as:

$\sigma = \sqrt{\frac{1}{m}\sum_i^M{(H - \mu)^2}}$

which is equivalent to the standard deviation. 

where $i$ is the $ith$ sample, thereby summing all features.

Essentially BatchNorm is computed **feature-wise**, for all input features to a given neuron, we compute the mean and variance and then an independent $\sigma$ and $\mu$ for each neuron which receives the feature.

> Note that BatchNorm is meant more for mini-batch or full-batch descent, and when the sample size is small, nearing 1, does not work well and can break easily. You wouldn't get a variance / mean estimate for the needed sample size for the mean and variance calculation to yield any meaningful information. CNNs *can* differ, read more below.

You can initialize $\sigma$ to $1$ while $\mu$ can be initialized to $0$.

You must also replace the bias parameter, $\beta$ in your model. Otherwise, the learnt $\mu$ will have no effect.

Ultimately, this reduces the covariate shift as we regularize the parameter updates to a fixed magnitude, therey reducing the disparity in their values as they train. This stabilizes training and minimizes the degree of stochasticity in the weight update.

You also end up reducing the risk of overfitting as the parameters are smoothed out, decreasing the magnitude of **bias** that they might have towards a given feature. THey all contribute rather than a handful overfitting to a **single feature.**

Note that during testing, you don't use the last set values for the parameters, $\sigma$ and $\mu$. Instead during training, you keep an exponentially weighted average (or leaky average) of the parameters $\sigma$ and $\mu$ and use the final averaged value gotten during training as the parameters for inference. The ewa, uses a hyperparameter, just like first and second moments of Adam, $\beta$, to denote how smooth we want to average the past statistics, $\sigma$ and $\beta$.

While BatchNorm works for larger batch sizes, things tend to break down when the batchsize becomes smaller. This is as for small batch sizes, say on the extreme end of SGD, a batch size of $1$, you cannot effectively compute the mean and variance for the entire dataset.

In fact, a batch size of $1$ would have a variance of $0$ and a mean as itself.

Of course, even with batch sizes of $5$ to $10$, if the entire full-batch size is much larger, then $\mu$ and $\sigma$ wouldn't hold meaningful information about the full-batch statistics.

In situations with low computational power, this becomes impractical, because you might not have enough memory to have a batch size where you can effectively compute the $\mu$ and $\sigma$ that effectively represent the dataset and the respective activations of the model to the dataset.

In CNNs, this issue isn't as prominent, as a CNN aims to build a hypothesis for the optimal weights without fully connected layers, and instead through convolutions. Then of course, we don't have a unique weight for each feature, rather we have $\text{weights} < \text{input features}$, as we're convolving the same set of weights over different spatial locations in the input feature map (or image in the input layer).

Thereby, given that we have multiple weighted sums / activations per channel ( which can be analagous to having multiple outputs per layer for multiple smaples ), we have enough data to construct a distribution of our data via $\sigma$ and $\mu$ in BatchNorm.

The values of $\sigma$ and $\mu$ are constructed, as a single value per channel, rather than as a single value per neuron in a fully connected network / layer.

Hence BatchNorm may still work effectively, if the input features to a given $lth$ layer has enough spatial area to convolve upon. 

Then the effectiveness of BatchNorm for the $lth$ layer can also be dependent on the size of your kernel for the $lth$ layer of your ConvNet. The smaller kernel size you have, the more weighted sums / activations you will have, and thereby the more data you will have to construct $\sigma$ and $\mu$. 

### Layer Norm

> [Paper](https://arxiv.org/pdf/1607.06450)

In concern to the above issue, when we have a batch size of $1$ and do have kernel sizes that are large enough such that we don't have enouguh activations to compute a meaningful $\mu$ and $\sigma$, **Layer Norm** can be a solution.

**LayerNorm**

It works exactly like BatchNorm, except it's applied to a single observation at once.

This time, rather than computing

```math
A_i' = \frac{A_i - \mu_i}{\sigma_i}\\[3mm]
\mu = \frac{1}{m} \sum_i^M {A_i}\\[3mm]
\sigma = \sqrt{\frac{1}{m}\sum_i^M{(H - \mu)^2}}
```
where $i$ is the $ith$ sample,

we can instead do the same thing but compute $\mu$ and $\sigma$ for all features in the **single** input activation.

Then here:

```math
A_j' = \frac{A_j - \mu_j}{\sigma_j}\\[3mm]
\mu = \frac{1}{m} \sum_j^M {A_j}\\[3mm]
\sigma = \sqrt{\frac{1}{m}\sum_j^M{(A_j - \mu)^2}}
```

$j$ is the $jth$ feature, we're summing over all features of the output activation of the $l - 1$ layer, across all samples.

Then, in a fully connected feed forward neural network (big word), at the $lth$ layer, we end up with $i$ unique values of $\mu$ and $\sigma$, each for each sample.

These values of $\mu$ and $\sigma$ are used for all neurons at the $lth$ layer. Each $l+n$ or $l-n$ ($n$ is any int) layer has it's own unqiue set of $\mu$ and $\sigma$.

In ConvNets, LayerNorm would be computed as the same, summing over $j$ features in the input feature map. We get a unique set of $\sigma$ and $\mu$ for each input channel, overcoming the issue with BatchNorm. No need to worry about limiting your kernel size any longer.

For inference, just like BatchNorm, we can run a leaky average of $\sigma$ and $\beta$ to use as parameters during forward passes.

### Group Normalization

> [Group Normalization](https://arxiv.org/abs/1803.08494)

Group Normalization is akin to Layer Normalization, with the difference that instead of taking the $\mu$ and $\sigma$ of the entire set of features for the $ith$ sample, we instead take the $\sigma$ and $\mu$ for a subset of the sample.

This allows is to compute more specific statistics based on the spatial features of each individual sample, allowing us to make more specific hypothesis about the optimal values of $\mu$ and $\sigma$. Thereby, the issue with the $\text{batchsize} =1$ is even further mitigated as we're able to more precisely compute the $\sigma$ and $\mu$, for each sample.

Ultimately, we take in the spatial information of each feature map into account and are able to compute statistics based on the ***specific*** spatial structure of each feature map. This allows more specific construction of the optimal $\beta$ and $\mu$, mitigating the covariate shift my an order of magnitude that is influenced by the size of your gorup, $g$.

It's defined as

```math
x_k' = \frac{x_k - \mu_k}{\sigma_k}\\[3mm]

\mu_k = \frac{1}{m} \sum_{k \in S_i} x_k\\[3mm]

\sigma_k = \sqrt{\frac{1}{m} \sum_{k \in S_i}^K (x_k - \mu)^2} \\[3mm]

```
where $k$ denotes the $kth$ group in the set of groups, $S_i$.

Each $x_k$ comes from $A_i$, the given feature map / channel. $x_k$ is a group instance from $A_i$, there are $g$ total groups, a hyperparameter that needs to be tuned.

Ultimately, the multiple $x_k'$s that are computed then form the final group normalized activations as $\hat{A_i}'$.

## Other Notes

- The choice of normalization depends on your batch size. If you have batch size = 1, it might be optimal to use group norm. If you have batch size = 5, perhaps Layer Norm might be the answer. For larger batch sizes, say 1000+, it makes more sense to use BatchNorm, rather than LayerNorm or GroupNorm. Having more precision means more variance, your mean and variance parameters will not generalize for all samples.