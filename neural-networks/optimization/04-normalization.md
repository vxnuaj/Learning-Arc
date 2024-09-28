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

Thereby, the gradients for a given neural network, aren't as large and the magnitude of the update for a given $\hat{y}$ becomes more predictable and less stochastic. It scales down the weight update, via a learnable distribution of activations for a given layer, $\gamma$ and $\mu$, where $\gamma$ is the variance of the activations and $\mu$ is the mean.

Given an activation matrix for the given $i$th layer, $A_i$, you can comute the normalized activations as the $z$-normalized matrix:

```math
A_i' = \frac{A_i - \mu_i}{\gamma_i}
```

$\mu$ and $\sigma$ are learned via gradient descent, exactly like any other parameter in the model is.

$\mu$ is constructed from $A_i$, shape $(\text{samples} \times \text{activations})$, by summing over the activations over all samples as:

$\mu = \frac{1}{m} \sum_i^M {A_i}$

while $\gamma$ is constructed in the same manner, in terms of axis operation, but computed as:

$\gamma = \sqrt{\frac{1}{m}\sum{(H - \mu)^2}}$

which is equivalent to the standard deviation. 

> Note that BatchNorm is meant more for mini-batch or full-batch descent, and when the sample size is small, nearing 1, does not work well and can break easily. You wouldn't get a variance / mean estimate for the needed sample size for the mean and variance calculation to yield any meaningful information.

You can initialize $\gamma$ to $1$ while $\mu$ can be initialized to $0$.

You must also replace the bias parameter, $\beta$ in your model. Otherwise, the learnt $\mu$ will have no effect.

Ultimately, this reduces the covariate shift as we regularize the parameter updates to a fixed magnitude, therey reducing the disparity in their values as they train. This stabilizes training and minimizes the degree of stochasticity in the weight update.

You also end up reducing the risk of overfitting as the parameters are smoothed out, decreasing the magnitude of **bias** that they might have towards a given feature. THey all contribute rather than a handful overfitting to a **single feature.**

Note that during testing, you don't use the last set values for the parameters, $\gamma$ and $\mu$. Instead during training, you keep an exponentially weighted average (or leaky average) of the parameters $\gamma$ and $\mu$ and use the final averaged value gotten during training as the parameters for inference. The ewa, uses a hyperparameter, just like first and second moments of Adam, $\beta$, to denote how smooth we want to average the past statistics, $\gamma$ and $\beta$.

### Layer Norm

BatchNorm sought to reduce the dependencies of previous layers on the later layers by normalizing the outputs of each layer to a fixed range, denoted by the variance ($\gamma$) and mean ($\mu$). 

While this works for larger batch sizes, things tend to rbeak down when the batchsize becomes smaller. This is as for small batch sizes, say on the extreme end of SGD, a batch size of $1$, you cannot effectively compute the mean and variance for the entire dataset.

In fact, a batch size of $1$ would have a variance of $0$ and a mean as itself.

Of course, even with batch sizes of $5$ to $10$, if the entire full-batch size is much larger, then $\mu$ and $\gamma$ wouldn't hold meaningful information about the full-batch statistics.

In situations with low computational power, this becomes impractical, because you might not have enough memory to have a batch size where you can effectively compute the $\mu$ and $\gamma$ that effectively represent the dataset and the respective activations of the model to the dataset.

**Layer Norm is the solution to this issue**