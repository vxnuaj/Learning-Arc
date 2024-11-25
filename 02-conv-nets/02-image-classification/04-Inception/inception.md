# Inception

<img src = 'https://media.geeksforgeeks.org/wp-content/uploads/20200429201549/Inceptionv1_architecture.png'>


## Going Deeper with Convolutions

**Initial Thoughts**: 

It's interesting to see how Inception uses multi-sized $\mathcal{K}$. For intuition, I think it'll help a given layer learn representations of different sized receptive fields, such that subsequent layers have "sight" into different ways of how the spatial structure is defined for a given input feature.

You combine this with more inception blocks, and hierarchically, each layer gets a more varying / descriptive insight into how an input feature map is constructed, spatially.

You're able to capture **both** fine-grained details and broader higher-level larger structures of a feature map with the inception module, building multi-scale features.

At a given $l$ layer, it has multi-sized receptive fields, depending on the branch you consider as each branch has different sized $\mathcal{K}$

### Definitions

**NiN:** *Network in Network*, by Lin et al

**Polyak Averaging**: Track an exponentially weighted average of the weights and use those weights during inference / testing.

```math

\bar{\Theta}_t = \alpha\bar{\Theta}_{t - 1} - (1 - \alpha)\Theta_{t}

```

where $\Theta_t$ is the parameter in question, $\bar{\Theta}_t$ is the averaged parameter, and $t$ is the current iteration.

The averaged weights are used define the model during inference. If the model is oscillating between the parallel points, surrounding the optima, applying a weighted average can help move the parameters to the optimal set of weights for the model.


### Introduction

- Progress in Computer Vision (2012 - 2014) hasn't been purely bigger data and GPUs but also more efficient yet accurate algorithms
- With on-device inference, on phones and other embedded computing, power and memory efficiency as extremely important
- Inception was built with a computational budget of 1.5B mulitply-adds during inference.
- Outperforms SOTA (as of 2014) in ILSVRC.

### Related Work

- Inception is a 22-layer model
- Inception makes use of $1 \times 1$ convolutions, to perform channel-wise parametric pooling, such that for the next layer, computational complexity is reduced while still retaining valuable information regarding relationships of a given feature at the $i, j$ position of the feature maps, across channels.
  - Allows for increasing depth and larger receptive field of the model.
- For Inception's object detection sumission to ILSVRC, they used Selective Search, just like R-CNN to generate bounding box region proposals

### Motivation & Considerations

- The best means to improve model performance is to increase their size, both width and depth, primarily depth.
  - Unfortunately this increases computational complexity.
  - This also increases the need for a larger dataset to prevent overfitting, which can be expensive to gather.
- A means to reduce network size is to introduce sparsity, by conjecturing that a generalized accurate hypothesis of the spatial structure can be build with smaller sets of parameters in $\mathcal{K}$. (This is why Inception uses multiple branches of different sizes, *why have a single branch of $5 \times 5$ convolutions when you can use two branches of $1 \times 1$ and $5 \times 5$, with the same amount of output channels as the former after concatenation?* **You're still able to learn important features despite having less parameters (given a smaller $1 \times 1$ conv).**
- To construct the optimal construction (without redundant weight sizes), you can analyze the correlation statistics of different regions in a given input, $X$, and construct kernels of sizes related to the dimensions of highly correlated regions.
- For earlier layers, a given $\mathcal{K}$ is trained to detect low-level features, such as edges, corners, etc. These low level features aren't abstract and are more locally defined, hence a given activation from $\mathcal{K}$, will likely be similar or correlated to other activations accross channels of the given activation of $\mathcal{K}$. 
  - Such then is the case that you can use as $1 \times 1$ convolution to sumamrize the highly correlated activations, in a parameterized manner, with a non-linearity to retain the valuable information of the feature map at the specific $i, j$ position.
  - As you go deeper into the convnet, the extracted edges and corners will begin to represent more spatially spread out clusters of important features (given by correlation statistics) such that you'll be able to learn them through larger sizes of $\mathcal{K}$, such as $5 \times 5$ convolutions. 
  - Thereby, as you get deeper into the model, you want to be able to increase the ratio of larger sized $\mathcal{K}$ ($3 \times 3$ & $5 \times 5$) relative to the $1 \times 1$ $\mathcal{K}$.

- The issue with increasing the ratio of $3 \times 3$ and $5 \times 5$ convolutions as we get deeper is that the number of output channels can continue to grow large. So drawing inpsiration from NiN, they use $1 \times 1$ convolutions to perform depthwise parametric pooling, such that they compress the amount of channels but aim to continue retaining information (via parameters and a non-linearity after each $1 \times 1$ conv.).

### GoogLeNet

<div align = 'center'>
<img src = 'https://media.geeksforgeeks.org/wp-content/uploads/20200429201421/Inception-layer-by-layer.PNG' width = 500>
<img src = 'https://media.geeksforgeeks.org/wp-content/uploads/20200429201549/Inceptionv1_architecture.png'>
</div>
<br>


> #3x3 reduce and #5x5 reduce columns indicate the amount of 1x1 conv filters used prior to a given convolution (either 3x3 of 5x5, respective to the column label.)

- All convolutions use $\text{ReLU}$ including those within the inception modules
- The network takes in $224 \times 224$ images.
- They use linear layer after global average pooling isntead of $1 \times 1$ convolutions, simply for purposes of allowing fine tuning, but still needed the use of dropout to avoid overfitting.
- The large depth of the network introduced the risk of vanishing gradients, therefore they used auxiliary classifiers after certain stages of the architecutre.
  - The auxiliary classifiers were used after the $3$rd and $6$th inception module.
  - They were comprised of:
    1. $5 \times 5$ average pooling layers with stride $3$
    2. A $1 \times 1$ convolution with $128$ filters for channel reduction
    3. $\text{ReLU}$ after each conv and fc layer besides output
    4. an FC layer with $1024$ neurons and Dropout with $p = .70$.
    5. an FC layer with $1000$ neurons and Softmax Activation
- Momentum with $\beta = .9$.
- Decreased learning rate by 4% every 8 epochs.
- Polyak averaging was used for the final model used at inference.
  
### Key Insights / Closing Thoughts

Earlier layers should be optimzed to learn local features, wuch as edge and corners, and slowly increase the receptive field over time.

- The deeper your model is, the more parameter efficient and accurate it becomes (hierarchical non-linear feature extraction)
- The correlation statistics of a given input image denote that low-level features tend to be concentrated in local regions (edges and corners are locally correlated and localized phenomena $\rightarrow$, they can be learnt via $1 \times 1$ convs)
  - Your receptive field remains small when you initially apply the $1 \times 1$ convs.
- As your model learns local features, it will construct feature maps where the **important** edges and features are more well defined such that they begin to define larger correlated regiosn of features, that can be learnt via large convolutions, such as $3 \times 3$ or $5 \times 5$ convolutions.
  - The receptive field begins to slowly grow as we continue to introduce the larger convs.

## Rethinking the Inception Architecture for Computer Vision

### **Abstract**

- Large-scale convolutional networks are bringing about improvements due to larger parameter count and deeper models, despite the cost of computational complexity.
- Despite this, the use of efficient models is important for edge devices such as mobile phones.
- They look at means to scale up networks through factorized convolutions and regularization, through inception v2/v3

### **Introduction**

- AlexNet and VGGNet have been extremely successful for improving the role of ConvNets, but come with the price of extreme computational cost, given the oncoming of big data needed to effectively train a model.
- There are general principles and guidelines that can be used to optimze neural network design such that efficiency is maximized while performance is retained -- hence optimizing performance.

### **General Design Principles**

- Avoid extreme amounts of dimensionality reduction. Reducing dimensionality with a high order of magnitude that is too high can hinder the amount of valuable and meaningful information (given by amount of retained correlation / variance) that is passed onto the next layer of a neural network. Make sure that the $n$-dimensional feature space gradually decreases over time.

> Do so while also increasing the amount of output feature maps, in context of convnets.

- Higher $n$-dimensional representations of a given input allow for a convnet to learn more seperable, distinguishable feastures. The convnets will train faster.

> A model is able to learn unique features, given that we have more activations coming out of a given $l$ layer, and provide more accurate results **quicker**.

- Spatial aggregation (via $1 \times 1$ convs), can be used effectively prior to convolutions of larger parameter size, $3 \times 3$ for example, without losing valuable information. This may be due to the fact that adjacent feature maps may have highly correlated values, such that the representational power of having a multi-channel input to a $3 \times 3$ conv, might be redundant and thereby $1 \times 1$ convs are able to capture the needed representations while discarding those thast aren't needed 

> The ideal # of output ch. probably needs to be tuned like a hyperparam, to determine the optimal amount of dimensionality reduction one needs. You can probably figure this out by analyzing the Covariance matrix of the activations of the incoming inputs to the $lth$ layer.

- Optimal improvement to a convnet can be done by **both** increasing the width and depth of the network, in **parallel**.

### Factorizing Convolutions

#### 3.1 Factorization into Smaller Convolutions

- Adjacent feature maps, which are highly correlated, can be summarized into a smaller set of feature maps, removing redundancy, via $1 \times 1$ convolutions (also known as Depthwise Parametric Pooling).
  - The high correlation implies that each $ith$ feature map in the output $A$ may have redundant features thast aren't very indistinguishable, hence we can reduce them via $1 \times 1$ convolutions, safely.

- You can easily factorize a convolution by increasing it's depth while decreasing kernel sizes. For a $5 \times 5$ single layer convolution, you'll have the same receptive field as 2 layers of $3 \times 3$ convolutions, with a decreased count of parameters and improved feature representations due to intermediate non-linearities.
- While reducing the $5 \times 5$ conv to 2 $3\times 3$ convs, without a non-linearity might make intuitive sense at first, to retain the linearity, empirically, introducing a $\text{ReLU}$ introduces better results as we capture more meaningful non-linear information.

#### 3.2 Spatial Factorization into Assymetric Convolutions

- If convolutions greater than $3 \times 3$ can reduced into multiple $3 \times 3$ convolutions, then higher dimensional $5 \times 5$ convs might not be useful due to increased parameter size.

  > Such that the inception module (1x1 -> 5x5 branch) can be improved so that it's composed of 1x1 -> 3x3 -> 3x3

- $3 \times 3$ convolutions can then be further reduced into $2 \times 2$ convolutions, but also into assymetric $n \times 1$ convolutions (or $1 \times n$)
  - A set of convolutions as $3 \times 1 \rightarrow 1 \times 3$ is the same as a $3 \times 3$ convolution, we still get a scalar as $A$.
  - Requires less operations and parameters than $2 \times 2$ convolutions
  - Requries less parameters than a $3 \times 3$ convolution, but with a drawback that we aren't able to detect diagonal features...
  
  > but somehow we're still able to get away with it.

- Theoretically you could replace any $n \times n$ convolution and replace it with a $n \times 1$ convolution... but in practice, it doesn't work for earlier layers (could likely be that earlier features include diagonals, which are important to extract for a given neural network and it's dataset).

### Utility of Auxiliary Classifiers

- Aux Classifiers were used to introduce auxliary losses and gradients such that we have improved gradient flow, reducing the vanishing gradient problem
- It was found that during the beginning of training, auxiliary classifiers didn't help much, until the end of training.

  > This can be expected as towards the end, vanishing gradients can become an issue as your gradient begins to approach lower values, less than $1$.

- In the original GoogLeNet, the first auxiliary classifier was conjectured to help train earlier layers to learn to represent low-level features, given that earlier layers would have a greater issue of vanishing gradients.
- It was found that removing the first auxiliary classifier yields near equivalnet performance as with.
  - The aux classifier can then be seen as a mere regularizer, helping reduce overfitting by introducing greater error, such that the weights for a layer don't become too large.
    
    - It's supported by the fact that that the main classifier performs better if the aux classifier is batch normalized or has a dropout layer, as mitigating the magnitude of gradietns from the aux classifier can produce smaller updates such that the parameters in the main classifier don't become as large.

### Efficient Grid Size Reduction

- We can typically use pooling to reduce the dimensions of a given output feature map, say we go from $d \times d \rightarrow \frac{d}{2} \times \frac{d}{2}$.
- This typically reduces the expression feature representations of a given layer such that it's wise to use a larger convolution prior, to effectively capture features spatially and depthwise.
  - Say we want to arrive to $\frac{d}{2}^2$ size with $2k$ filters, we'd need to apply $2ki$ convolutions (assuming $p = 1$) where $i$ is the number of input channels and $k$ is the output channels, and only then apply a pooling layer, say $2\times 2, s = 2$. The operation cost would then be $2d^2k^2$.
    
    - This introduces a high computational cost
  - Alternatively to reduce computational cost, we can apply pooling first and then apply a convolution, reuslting in $2\frac{d}{2}^2k^2$ operations, but we introduce a representational bottleneck where we lose model expressiveness.
  - Ultimately then, we can branch the layers into 2 branches, where a one is a pooling layer, $P$, and another is a conv layer, $C$, where $C$ provides half of the final filter bank size and $P$ provides the other half, to then be channel-wise concatenated at the end of the module.

### Inception-v2

- Typical $7 \times 7$ input convolution is factorized into $3 \times (3 \times 3 \text{ convolutions})$ (for the rest of the architecture, see paper at Table $1$)
- $3 \times$ Traditional Inception Modules
- Efficient Grid Reduction Layer | $35 \times 35 \rightarrow 17 \times 17$
- $5 \times$ Factorized Inception Modules (see fig 5.)
- Efficient Grid Reduction Layer | $17 \times 17 \rightarrow 8 \times 8$
- $2 \times$ $n \times 1$-type inception modules (see fig 6. )
- The quality of the network is relatively stable to variations in its architrecture as long as prior principles are kept (see section $2$)

### Model Regularization via Label Smoothing

- Given a one-hot encoded vector of label probabilities for a sample ($X$), $q$, we can comptue the cross entropy loss as $\mathcal{L} = - \sum log(p) q$, where $p$ is the softmax activation for a given $z$, $p = \frac{e^z_i}{\sum_i^K e^z_i}$
- Denoting $q$ as one-hot encoded vector where the index for ground-truth probability is $1$ and the rest are $0$ can cause overfitting in a model, as we essentially aim to let the model determine the absolute truth, with no uncertainty.
- Instead we can "smooth" the labels such that the ground truth is not $1$ and the incorrect labels aren't $0$.
  
```math

q(k | x) = ( 1 - \epsilon ) \delta_{k, y} + \frac{\epsilon}{K}

```

where 
- $\epsilon$ is the smoothing hyperparameter
- $\delta = \begin{cases}1, k = y \\ 0, k ≠ y\end{cases}$
- $\frac{\epsilon}{K}$ is a value drawn from a uniform distribution of such that $0$'s become evenly distributed, as the same value, when $\delta = 0$
- $k$ is the class index, indexing the one-hot encoded vector.


### Other

- Inception V3 = Inception V2 + Batch Normed Auxiliary Network


### Insights / Thoughts

- When using $1 \times 1$ convolutions, to reduce computational cost, it's important to consider the correlation statistics of the given input feature matrices.
  -  If they are highly correlated, then perhaps the the $1 \times 1$ convolutions will serve well in reducing redundancy.
- Spatial aggregation (via $1 \times 1$ convs), can be used effectively prior to convolutions of larger parameter size, $3 \times 3$ for example, without losing valuable information. This may be due to the fact that adjacent feature maps may have highly correlated values, such that the representational power of having a multi-channel input to a $3 \times 3$ conv, might be redundant and thereby $1 \times 1$ convs are able to capture the needed representations while discarding those thast aren't needed 

> The implication would be that we fully train a network and analyze the intermediate activations prior to the $3 \times 3$ convolutional layers. If they have a high covariance, it might be safe to retrain the model, this time with a $1 \times 1$ convolution layer before the $3 \times 3$. You might not even need to fully train the model to realize this. Just compute the co-variance matrices after intermeidate checkpoints.
>
> 1. Flatten each channel such that $H \times W \times C$ becomes $D \times C$, where $C$ is channel count.
> 2. Insert the new vectors as columns into a matrix, $X$.
> 3. Compute co-variance as $\frac{X^TX}{n - 1}$
> 4. If there is high-covariance, safe bet to introduce a $1 \times 1$ convolution without losing much representational power of your hidden activations.