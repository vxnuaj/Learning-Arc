# Normalization

### BatchNorm

For a given Neural Network, $f$, it's parameters, $W$, can have wildly shifting magnitudes over different layers. Over time, this shift could hamper the convergence of a neural network. 

Consider we have a fixed learning rate $\alpha$, for all layers.

If layer $l$ has a mangnitude of $10\times$ the size of $l-1$, the learning rate will result in enormous updates for $l$ but not for $l - 1$, such that learning can become unstable.

> Covariate Shift

Also, variable magnitudes in $W$ can indicate overfitting as larger $w_i$ relative to other $w_i$ can indicate that the neural network $f$, is more intricately modelling the function rather than landing on a general solution.

Batch Normalization solves the covariate shift while also regularizing the neural network

### Batch Norm for Conv Nets

Given a convolution output, $Z$, of shape $N, C, H, W$, we compute the batch statistics, $\mu$ and $\sigma$ (standard deviation) as:

```math

\mu_c = \sum_N \sum_H \sum_W Z_{n, h, w}
\\[3mm]
\sigma_c = \sqrt{\sum_N \sum_H \sum_W (Z_{n, h, w} - \mu)^2}
```

We sum over all spatial locations of $Z$ for channel sample $n$ and then compute the statistics over all $N$ samples. 

Each feature map, has it's own statistics, but it is shared across $N$ samples.

### LayerNorm for Conv Nets

Computed the same as batchnorm but with a subtle difference. Rather than normalizing over a set of samples, we compute statistics over channels as:

```math

\mu_c = \sum_C \sum_H \sum_W Z_{n, h, w}
\\[3mm]
\sigma_c = \sqrt{\sum_C \sum_H \sum_W (Z_{n, h, w} - \mu)^2}
```

This allows for us to make use of the feature representations using a single batch.

### Other

- Note that for a mini-batch size of $1$, BatchNorm still works as we still have spatial dimensions $H$ and $W$ to normalize over. Of course, most frameworks don't implement this feature, but mathematically, it still is computable.