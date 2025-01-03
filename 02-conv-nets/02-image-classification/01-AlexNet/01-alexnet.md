<img src = 'https://media.licdn.com/dms/image/D4E12AQER6q7Ke59psw/article-cover_image-shrink_600_2000/0/1657672566956?e=2147483647&v=beta&t=fs7GbtKgFL1hgTWMNkuGvq79-xofS-ax0hn7HYpOWok'>
<br>
<br>

# AlexNet

Prior to modern day Computer Vision, machine learning (not even dl yet), wasn't about the algorithm but rather about the pre-processing pipelines and feature extractors that were used to prepare a given dataset.

Kernel-methods and ensemble models reigned king, but weren't useful in terms of directly classifying a given dataset. What made those learning algorithms useful were the preprocessing steps and feature extraction pipelines applied to the original datasets, such that only then, they were able to "learn" meaningful information.

More modern deep learning models, such as AlexNet, were able to automate this feature extraction (via convolutional blocks), to learn hierarchical and increasingly complex representations of the original dataset (representation learning, where you learn the most important features that describe a dataset). 

### Big Data

- ImageNet (2009) $\rightarrow$ more data, cleaner data (images)
- More data $\rightarrow$ better generalization / accuracy for potential deep learning models

### Hardware

#### CPUs / GPUs

- **Clock Rate:** The rate at which a given processor completes a computation, measured in Hz, modern day processors are on the scale of GHz (billions of Hz)
- **Cores:** Smaller processing units within the chips, which directly complete a given computation. Many modern chips are multi-core, meaning they can complete multiple instructions in parallel. Cores have their own clock rate, a chip with 3 GHz as it's clock rate will have 3 GHz per core.
- **Threads**: Running multiple processes in parallel among the multiple cores of a chip.
- **Hyperthreading**: Running multiple processes in parallel among a single core of a chip.

CPUs are very expensive to build, many are able to execute a wide range of instructions with many uneeded features if only aiming for specialized computation.

GPUs have a large number of cores when compared to CPUs, with each core having a smaller clock rate relative to CPUs. It's the threading of the GPUs (and perhaps hyperthreading) with a higher order of cores that makes it much more faster than a CPU.

The first reason GPUs are much better than CPUs can be due to cost efficiency to achieve the same speed.

CPUs generally need a higher clock rate (given decreasing # of cores) to achieve the same speed that a GPU has with multiple cores.

The clock-rate, scales quadratically for an increasing clock-rate.

```math

P = C \cdot V^2 \cdot f

```

where:

- $P$ is the dynamic power consumption
- $C$ is the circuit capacitance
- $V$ is the voltage supply to the circit
- $f$ is the clock-frequency (clock-rate)

Note, $V^2$, such that as $f$ increases, we get a quadratic increase to $P$.

A GPU can have the same level of performance with a larger number of cores that each individual have a smaller $P$. 

If we have a CPU with a single core of speed $4 GHz$ and power $P$, we can use 4 GPU cores at $1 GHz$, such that $4 \cdot 1GHz = 4GHz$, yielding the same performance.

Given that $P$ increases quadratically, the $P$ for the CPU will end up costing more than $P_{GPU}$

GPU cores are also more efficient, as they're simpler. They don't have additional features and larger caches as CPUs have.

The clock-rate, scales quadratically for an increasing clock-rate.

```math

P = C \cdot V^2 \cdot f

```

where:

- $P$ is the dynamic power consumption
- $C$ is the circuit capacitance
- $V$ is the voltage supply to the circit
- $f$ is the clock-frequency (clock-rate)

Note, $V^2$, such that as $f$ increases, we get a quadratic increase to $P$.

A GPU can have the same level of performance with a larger number of cores that each individuall have a smaller $P$. 

If we have a CPU with a single core of speed $4 GHz$ and power $P$, we can use 4 GPU cores at $1 GHz$, such that $4 \cdot 1GHz = 4GHz$, yielding the same performance.

GPU cores are also more efficient, as they're simpler. They don't have additional features and larger caches as CPUs have.

This makes GPUs incredibly great for matrix multiplications and other neural network operations.

Given this, then came AlexNet!

### AlexNet 


<br>
<div align = 'center'>
<img width = 600 src = 'https://media.licdn.com/dms/image/D4E12AQER6q7Ke59psw/article-cover_image-shrink_600_2000/0/1657672566956?e=2147483647&v=beta&t=fs7GbtKgFL1hgTWMNkuGvq79-xofS-ax0hn7HYpOWok'>
</div>
<br>

Input images are $3 \times 224 \times 224$, 3 channels, and a height and width of $224$ pixels.

**Layer 1, $11 \times 11$ Conv, 96 channel output, stride 4**: 
- Given that we have higher resolution images ($224 \times 224$) pixels, we need a larger feature map to not limit the model to a purely local representation of the image (increasing receptive field right off the bat). 
- Stride of $4$, to downsample the input size from $224 \times 224$, allowing us to summarize information into important feature maps, by the representation of the Kernel $\mathcal{K}$. 
- $96$ channel output, allowing the model to learn $96$ unique yet important features for the input, given $96$ $\mathcal{K}$.

**Layer 2, Max Pooling Layer, $3 \times 3$, Stride 2**
- To reduce dimensions, by extracting the highest valued feature of a given $3 \times 3$ region for each of the $96$ output feature maps, by choosing the highest value feature 
- Stride of $2$ to further downsample to a smaller size

> Higher valued features typically corresponds to the most important learned feature for the given layer

**Layer 3, $5 \times 5$ Conv, 256 Channel Output, stride 2, padding 2**

- Smaller kernel than layer $1$, now at $5 \times 5$, given that we've downsampled the original image to $27 \times 27$ after the pooling layer. Yet large enough that we're able to get enough global context for the next set of features we want to extract.
- $256$ channel output to increase the amount of unique features $\mathcal{K}$ learns, for $256$ different $\mathcal{K}$.
- Padding of $2$ to retain dimensionality of the input and properly learn the edges of the feature maps through $\mathcal{K}$.

**Layer 4, $3 \times 3$ MaxPool, Stride 2**
- Reducing dimensionality of the $256$ output feature maps, extracting the most important learned features. 
- Stride of $2$ to further downsample.
- Adding a small degree of invariance, within pixel shifts between $1$ to $3$

**Layer 5, 6 | $3 \times 3$ Conv, 384 Output Channels, Padding 1**

- We have 2 layers of $3 \times 3$ convolutions, again with smaller sizes of $\mathcal{K}$ when compared to earlier layers, given the downsampled feature maps, but not so small such that we limit the model to purely local information
- $384$ output channels, to increase the count of unique features we learn to $384$ with $384$ total $\mathcal{K}$.
- Padding size of $1$, to properly integrate edges of the image into the learned representation of $\mathcal{K}$.

**Layer 7 | $3 \times 3$ Conv, 256 channel output, padding 1**:

- Keeping dimensionality with padding 1, but decreasing the channel count to 256, to summarize and reduce to the more important features.

**Layer 8 | MaxPool, $3\times 3$, Stride 2**

- Extracting the most relevant features within each $3\times 3$ region of the input feature map. Stride of $2$ to further downsample, by skipping over redundant regions of the feature map (probably didn't think there was useful information to be capture by overlapping MaxPool regions.)
- Adding a small degree of invariance, within pixel shifts between $1$ to $3$

**Layer 9, 10, 11**

- FC Layers, to use the extracted flattened features for the final clasification.
- Dropout at layers 9 and 10, to prevent co-adaptation of neurons, allowing for regularization and better generalization ($p = .5$)

---

## AlexNet Paper

<details><summary>Definitions</summary>

**Top-1 Error Rate:** The error rate (in % values) of which the correct class is not corresponding to a model's **most confident** (hence top-1) prediction, across a set of samples.

**Top-5 Error Rate:** The error rate (in % values) of which the correct class is not corresponding to a model's **top-5 most confident** (hence top-5) predictions, across a set of samples.

**Co-variance Matrix**: Covariance is the measure of linear association between two variables. A positive value indicates an increasing relationship between two variables. A negative value indicates a decreasing relationship between two variables. If two variables are independent, covariance will be 0, but a covariance of 0 does not mean independence. Co-variance does not denote strength of the relationship between two variables. Computed as an outer product. Elements on the diagonal tell you the variance. Off diagonal elements tell you the co-variance.

</details>

### **Abstract**

- AlexNet was trained on ImageNet1k, a dataset comprised of $1.281 \text{ million}$ images
- Sped up training via $max(0, z)$, the $\text{ReLU}$ non-linearity.
- Reduced overfitting via Dropout, $p = .5$
- Top-5 test error rate of 15.3% on the ImageNet 2012 Competition

<details><summary>On ImageNet1k</summary>

See images [here](https://github.com/EliSchwartz/imagenet-sample-images)

The ImageNet dataset was a $1.281$ million sample dataset, containing $1000$ total classes ([see here for each class label](https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt)) of objects derived from **WordNet**.

Each class has about between 700 to 1.3k samples, each class having varying amounts.

It's a subset of the entirety of the ImageNet dataset, which was ultimately used in the ImageNet LSVRC contest, to push the boundries of object detection.

</details>

### **Intro.**

- Curr. datasets were limited, small, and not useful for training large neural networks which are able to generalize on real-world data. 
- Curr. hardware was not specialized nor powerful enough to train large and deep neural networks.
- CNNs, are able to be useful for object detection, as they rely on the principle of locality (up to a certain degree) to accurately classify images, given that the pixel statistics for a given object that is desired to be detected, should not change despite any shift in its position relative to the entire image(making obj. detect. suitable via the Convolution operation with kernel, $\mathcal{K}$)

- They trained the largest convolutional neural networks on a subset of ImageNet (ImageNet-1k), and achieved the best result ever reported thus far.
- Did so with optimized versions of a 2D convolution and other convolution ops.
- The depth of the network was extremely important, removing a single layer, despite being only 1% of total model size, resulted in detrimental performance.
- Their model was purely limited by the training time they were willing to tolerate and speed of GPUs. No indication of performance improvement diminishing with scale of data and depth of the model.

### **Dataset**

- ImageNet-1k, yielded rectangular pictures, of varying dimensionality, $D$.
- Their CNN only accepted images of fixed dimensionality, $\hat{D}$
- To get from $D \rightarrow \hat{D}$, they rescaled the shorter side of the given images to $256$ and extracted the center of the image
- No other pre-processing besides mean-centering.


### **Architecture**

**ReLU Non-Linearity**
  - Avoids saturating $∂$'s when we have $z$ nearing $1$ or $0$ (for $\sigma(z)$, as for $\text{tanh}(z)$, it's between $1$ and $-1$)

**Multiple GPUs**!
  - AlexNet trained on Multiple GPUs (parallelization), allowing them to bypass the fact that their GTX 580's only had 3GB of VRAM (GPUs use VRAM to store their parameters into their cache during training, the bigger the VRAM is, the larger model you can train).
  - Thereby, the model was split into 2, half the model residing on one GPU and the other half on the second GPU.  On at layer $3$ and the $\text{FC}$ layers, did the GPUs communicate with each other for training.
    - Therefore, at $\text{CONV-3}, \text{GPU-i}$, the kernel $\mathcal{K}_{i3}$, used all input channels to extract features, giving $\mathcal{K}_3$ a **global context** of the sample
    - At the $\text{CONV-N}$ where $n ≠ 3$, a given kernel $\mathcal{K}_{in}$, used a subset (half) of the input channels (feature maps) which were extracted at earlier layers. Then a given $\mathcal{K}_{in}$ only learns from a subset of extracted features.
    - At the $\text{FC}$ layer, there's inherent GPU communication, as we need all neurons to be fully connected with all previously extracted features.

**Local Response Normalization**

```math

b_{x, y}^i = \frac{a_{x, y}^i}{(k + \alpha \sum_{j = \text{max}(0, \frac{i-n}{2})}^{\text{min}(N - 1, \frac{i + n}{2})} (a_{x, y}^j)^2)^{\beta}}
\\[6mm]
\text{wtf}
```

where

- $i$ and $j$ are the indices for a given kernel set, $\mathcal{K}$ 
  
  >(here, a kernel set are all filters that are used for a given single output. $\mathcal{K}$ denotes the set of multichannel filters for the single output channel.).

- $x, y$ denotes the position of the output feature map
- $k$ is the bias, ensuring the denominator doesn't become too small, ensuring the denominator doesn't become too small.
- $\alpha$ scales the sum of the squared activations, the higher it is, the more normalization.
- $\beta$ controls the degree of normalization, the higher it is, the more normalization you apply onto $a_{xy}^i$
- $n$ controls the size of the local channel neighborhood.
- The bounds for the summation is essentially to cover edge cases for the formula (negastive output channels, stopping at the alst valid output channel).

Essentially, $\text{LRN}$ takes an activation at the $x, y$ position of the output, and normalizes it over a given neighborhood (hence **LOCAL** response normalization).

This is extremely useful when dealing with assymetric activations, where a given activation for a given $\mathcal{K_i}$ is extremely larger than other activations at neighboring $\mathcal{K_i}$. 

In the case of $\text{ReLU}$, given that it's an unbounded activation function on it's positive domain, activations can become extremely large.

This can easily bring about overfitting (similar to reducing co-adaptation of neurons), as some neurons are "lazy" (low activation values) are heavily reliant on those with higher activation values.

$\text{LRN}$ mitigates this issue by normalizing an activation at a given $\mathcal{K}_i$ with respect to activations at neighboring $\mathcal{K}_i$'s.

The reason this works is as we reduce the scale (decreasing $\mu$) but maintaing the variance ($\sigma$), hence we reduce the difference of values in individual activations but maintain the information (on a smaller scale).

- The ordering of $\mathcal{K}$'s is arbitrary, determiend prior to training.

> *(REAL NEURO) This is similar to **lateral inhibition** within biological neural circuits, where a given activation of a neuron reduces the strength of activation of neighboring neurons. In a sense, neurons 'compete' with each other, to increase the overall "sharpness" of the received signal by reducing the signal to noise ratio.*

$\text{LRN}$ layers in AlexNet were only applied on the first 1st and 2nd convolutional layers. Ensuring that within the first extracted features, every single feature is considered. It is not as important in later layers, as we ultimately want to extract **important** features in those layers.

This is generally a good rule of thumb (though can be volatile for other architectures, depending on receptive field, size of $\mathcal{K}$, etc).

**AlexNet used hyperparameters, $k = 2$, $n = 5$, $\alpha = 10^{-4}$, $\beta = .75$**

**Pooling**

- Used overlapping max-pooling layers, found that overfitting was mitigated.

**Overall Architecture**

- Maximized SoftMax probability objective (negative log likelihood), equivalent to maximizing the average $log$ likelihood over training samples.
- $\mathcal{K}$ in 2nd, 4th, and 5th laeyrs are connected to feature maps which reside only on the same GPU.
- $\text{LRN}$ layers were only applied on the first 1st and 2nd convolutional layers.
    - Ensuring that within the first extracted features, every single feature is considered.
    - Not as important in later layers, as we ultimately want to extract **important** features in those layers.
- Max-pooling layers after the 1st and 2nd layers as well as the 5th convolutional layer.
- $\text{ReLU}$ is applied to every hidden layer

### Reducing Overfitting

- Introduced transformations via translations and horizontal reflections
- Used PCA-based color augmentation

<details><summary>More on PCA-based color Augmentation</summary>

$\text{PCA}$ based color augmentation plays an important role in computer vision to introduce a set of distortions to the RGB values of a given image, to allow your model to learn from a variety of data.

Alongside the other data augmentations (flipping, translations, rotations, etc), $\text{PCA}$ based color augmentation introduces varying changes to the RGB values of your images, such that a given ConvNet has a greater variety of training data to learn from.

Given a sample, $X$, with $3$ color channels, $\mathcal{RGB}$, of shape $c \times n \times n$, we can $c$ principal components and the $c$ eigenvalues ($\lambda$) which correspond go those $c$ principal components. 

An individual pixel is denoted as:

```math

\mathcal{P}_{ij} = [\mathcal{R}_{ij}, \mathcal{G}_{ij}, \mathcal{B}_{ij}]

```

and the entire set of pixels is denoted as:

```math

\hat{X} = \begin{bmatrix} - P_{11} - \\ . \\ . \\ -\mathcal{P}_{IJ} \end{bmatrix}

```

When we perform $\text{PCA}$ on $\hat{X}$, we derive the principal components from the determinants of the co-variance matrix of $\hat{X}$, denoted as $C_{\hat{X}}$, which is computed as an outer product of $\hat{X}$, such that $C_{\hat{X}}$ is a $3 \times 3$ matrix.

```math

C_{\hat{X}} = \frac{\hat{X}^T\hat{X}}{IJ}

```

where $IJ$ is the total count of pixels in the image.

The co-variance matrix tells us the variance and the co-variance of different color channels for $X$. Elements on the diagonal are simply $\text{Var}(\mathcal{R}_{ij})$, $\text{Var}(\mathcal{G}_{ij})$, $\text{Var}(\mathcal{B}_{ij})$, given that we compute as an outer product and ultimately turn out to simply being the variance.

Elements on the off-diagonal are the co-variance of the corresponding elements that were multiplied together.

For an **RGB** image, the $C_{\hat{X}}$ turns out to be a $3 \times 3$ matrix, and therefore when we compute $\text{PCA}$ via eigendecomposition, we receive back $3$ principal components and $3$ eigenvalues which correspond to them.

So, to compute $\text{PCA}$

1. Find the covariance matrix of $\hat{X}$ as $C_{\hat{X}} = \frac{X^TX}{n}$
2. Find the eigenvalues and eigenvectors of $C_{\hat{X}}$, such that $C_{\hat{X}} = P\Lambda P^{-1}$ 

The eigenvectors of $C_{\hat{X}}$ correspond to the $3$ principal components alongside the $3$ eigenvalues ($\lambda$).

Now that we have the $3$ principal components ($p_i$) and $\lambda$'s, we can perform image augmentation on each pixel as follows:

```math

\vec{\text{aug}} = [p_1, p_2, p_3][\alpha_1\lambda_1, \alpha_2\lambda_2, \alpha_3\lambda_3]^T
\\[3mm]
\text{where }\vec{\text{aug}} \in \mathbb{R}^3
```

> this is a dot prod, not element wise. the first factor is a $3 \times 3$ matrix of principal components!!

```math
\mathcal{P}_{ij}^{aug} = \mathcal{P}_{ij} + \text{aug}^T
\\[3mm]
\mathcal{P}_{ij}^{aug} = [R, G, B]_{ij} + [R_{\text{aug}}, G_{\text{aug}}, B_{\text{aug}}]

```

where $\alpha \sim \mathcal{N}(\mu = 0, \sigma = .1)$, $p_i$ are the eigenvectors.

This preserves color variance, as the magnitude of each $\lambda_i$, relative to other $\lambda_i$ denotes the amount of variance that we aim to capture / store amongst different pixel $\in \mathcal{P}_{ij}^{aug}$.

The dot product with $\alpha_i \lambda_i$ scales the augmentation such that it aims to preserve the variance across all pixels pixels, influenced by the magnitude of $\lambda_i$.

$\alpha$ is purely drawn stochastically from a Gaussian distrinution for randomness in the augmentations.

More in-depth, the variance of each individual principal component ($\text{Var}(p$)) tells you the amount by which the direction of $p$ captures the variance of the original set of pixels.

The principal component that corresponds to the largest eigenvalue captures the most variance. \
The principal component that corresponds to the second-largest eigenvalue captures the second-most variance.

Each individual value within a given $p_i$ tells you how much each value in the original vector contributes to the new direction denoted by $p_i$. The larger a given $ith$ value of the $p_i$ is, the more contribution or strength the $ith$ value in the original vector had, of course all relative to other values in the pixel vector, $\mathcal{P}_{ij}$.

Then, given the dot product of

```math

\vec{\text{aug}} = [p_1, p_2, p_3][\alpha_1\lambda_1, \alpha_2\lambda_2, \alpha_3\lambda_3]^T
\\[3mm]
\vec{\text{aug}} = [R_{\text{aug}}, G_{\text{aug}}, B_{\text{aug}}]

```

we're capturing the contribution strength of each pixel value, $R$, $G$, and $B$ respectively across the entire image (given that we perform $\text{PCA}$ on the entire set of pixels) with an added multiple of $\alpha_i$ such that the augmentation vector, $\vec{aug}$, adds a slightly stochastic augmentation to each pixel in the image.

This was the way it was computed in AlexNet.
</details>

- Used dropout with $P = .5$, such that half the neurons in the FC layers get zeroed out per forward pass of the training phase.
    - Did not use inverted dropout.

### Details of Learning

- Momentum ($\beta$) = $.9$
- Weight decay $= .0005$
  - Interestingly works not only as a regularizer but also as a means to faster convergence. Minimizing the magnitude of the weights leads to a smoother loss surface, reducing the probability of getting stuck at a local minima.
- $W ~ \mathcal{N}(\mu = 0, \sigma = .01)$
- $B = 1$ for layers 2, 4, 5 Conv Layers and all FC layers, to accelerate training as $\text{ReLU}$ can have a positive, non-zero gradient. For all other, $B = 0$
- Learning rate was initialized to $.01$ and divided by $10$ when the validation error stopped improving with training error.
- Trained for 90 epochs of 1.2 million images (minibatched to 128 samples)

Rather than computing accuracy of the model (or loss), another way to gauge accuracy is to compute the euclidean distance between 2 of the outputs of the last layer prior to the softmax layer. If the two outputs have a small euclidean distance and the inputs were of same class, then it's likely the model is performing well.

- It's inefficient but you could train an autoencoder to compress the output vectors and then compute the $L2$ distance.

> Could use MSE as a metric but retain Cross-Entropy as a loss function.

### Other Notes

To compute total memory footprint for a given model, count the number of parameters:

- $W_{size}$
- $B_{size}$

for all layers, alongside their pre-activations and activations:

- $A_{size}$
- $Z_{size}$

Identity the numerical format of your params and activations (float16 vs float8 vs etc).

- Float16 has 4 bytes
- Float8 has 2 bytes
- etc has $X$ bytes

call this value $X$.

Total memory footprint is $M$ in bytes where:

```math

M = X(W_{size} + B_{size} + A_{size} + Z_{size})

```

This does not account for $∂$ values nor other parameters.

**Brightness** is pixel intensity while **contrast** is the difference (scale) between dark and light pixels.

---


As we downsample, we increase channel count...? 

Such that if we only downsampled but kept channel count constant, we'd reduce the features / important information of the image. Downsampling, reduces the size of the feature maps, but increasing channel count as we get deeper increases the count of **important** features we learn which are all of a smaller dimensionality.