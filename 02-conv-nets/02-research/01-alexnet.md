<img src = 'https://media.licdn.com/dms/image/D4E12AQER6q7Ke59psw/article-cover_image-shrink_600_2000/0/1657672566956?e=2147483647&v=beta&t=fs7GbtKgFL1hgTWMNkuGvq79-xofS-ax0hn7HYpOWok'>
<br>
<br>

- [ ] Local Response Normalization (Read Paper First, get to it as you get there.).

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

### AlexNet Paper

<details><summary>Definitions</summary>

**Top-1 Error Rate:** The error rate (in % values) of which the correct class is not corresponding to a model's **most confident** (hence top-1) prediction, across a set of samples.

**Top-5 Error Rate:** The error rate (in % values) of which the correct class is not corresponding to a model's **top-5 most confident** (hence top-5) predictions, across a set of samples.

</details>


**Abstract**

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


---


### Other

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

---


As we downsample, we increase channel count...? 

Such that if we only downsampled but kept channel count constant, we'd reduce the features / important information of the image. Downsampling, reduces the size of the feature maps, but increasing channel count as we get deeper increases the count of **important** features we learn which are all of a smaller dimensionality.