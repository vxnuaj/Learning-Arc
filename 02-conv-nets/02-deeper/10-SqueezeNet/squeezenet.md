# SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size

### Introduction and Motivation

For a given task, there are multiple ConvNet Architectures that can achieve similar accuracy.

Given equivalent accuracy, smaller ConvNets have benefits of:

- More efficient distributed training -- communication overhead is proportional to the number of params.
- Less overhead, when exporting new models to edge devices -- Tesla's over the air updates for their Autopilot may require large data transfers, given modern ConvNets, smaller models make communication and updates more feasible.
- FPGA's often have $< 10$ MB of on-chip memory with no off-chip memory. A sufficiently small model could be stored on the FPGA, without being bottlenecked by memory bandwidth.

### SQUEEZENET: PRESERVING ACCURACY WITH FEW PARAMETERS

**Strategy 1**: Replace $3 \times 3 \hspace{1mm} \mathcal{K}$ filters with $1 \times 1 \hspace{1mm} \mathcal{K}$

Given that $3 \times 3$ filters have a large count of paramaters, and a restricted budget for memory, most of them are discarded, opting for $1 \times 1$ instead.

**Strategy 2**: Decrease count of input channels to $3 \times 3$ Convolutions 

This is done via *squeeze layers* (see below).

**Strategy 3**: Downsample later in the network, such that ConvNets have higher $H \times W$ dimensional feature maps. 

This has the effect thaat the initially extracted features are **richer**, of higher quality, while remaining the same or similar dimensions.

You extract higher quality features, at a higher spatial dimensionality.

Then, later layers with larger $\mathcal{K_{\ell}}$ or $s$ and smaller $Z_{\ell}$ will be able to extract **richer** features.

### Fire Module

1. *Squeeze* Conv Layer
   1. $1 \times 1 \hspace{1mm} \mathcal{K}$ 
2. *Expand* Conv Layer (multibranched)
   1. $1 \times 1 \hspace{1mm} \mathcal{K}$  
   2. $3 \times 3 \hspace{1mm} \mathcal{K}$

$s_{1 \times 1}$ controls the width of the *squeeze* conv layer <br/>
$e_{1 \times 1}$ controls the width of the $1 \times 1$ convolution in the *expand layer*<br/>
$e_{3 \times 3}$ controls the width of the $3 \times 3$ convolution in the *expand layer*<br/>

$s_{1 \times 1} < (e_{1 \times 1} + e_{3 \times 3})$, to limit the count of input channels to the $3 \times 3$ convolution, as mentioned in **strategy 2**.

- Padding of the $3 \times 3$ Convolution is $1$.

### Architecture

<br/>
<div align =  'center'>
<img width = 900 src = 'https://i.sstatic.net/0pOi4.png'/>
</div><br/>

1. Initial $96$ Channel Convolution
2. 3 Fire Modules
3. Max Pooling, $s = 2$
4. 4 Fire Modules
5. Max Pooling, $s = 2$
6. Fire Module
7. Conv
8. Global AvgPool

- $\text{ReLU}$
- Dropout for the last Convolution Layer

$base_e = 128\\incr_e = 128\\pct_{3 \times 3} = .5\\freq = 2\\SR = .125$ (see below)

### Experiments

After applying model compression, Squeeze Net reaches .46 MB while still having AlexNet level accuracy.

### CNN MICROARCHITECTURE DESIGN SPACE EXPLORATION

$base_{e}$ is the number of filters in the *expand* layer of the first fire module, meaning $base_{e} = e_{1 \times 1} + e_{3 \times 3}$ for the $e_i$ in the first layer.

$incr_e$ is the incrementation to the number of expand filters at every fire module $i$.

$freq$ is the frequency of incrementing the total count of expand modules.

The total number of expand filters in the $i$th expand module is defined as:

```math

e_i = base_e + (incr_e \times \lfloor\frac{i}{freq}\rfloor)

\\

```

Given 10 filters at the first fire module, with an incrementation of $10$ every second module, and assuming we're currently at layer $4$, then:

```math
10 + (10 \times \frac{4}{2}) = 30 \text{ expand filters at layer i}
```

The total count of $e_{3 \times 3}$ is denoted as a $pct_{3 \times 3}$ of the original $e_i$ -- $e_{3 \times 3} = pct_{3 \times 3} \cdot e_i$, the important hyperparameter is $pct_{3 \times 3}$.

The rest of the remaining filters are denoted to be $e_{1 \times 1}$.

The squeeze ratio is the ratio of filters in the squeeze layers compared to the expansion layers. Then $s_{1 \times 1}$ = $SR * e_i$.

Squeeze Net has the following parameters:

$base_e = 128\\incr_e = 128\\pct_{3 \times 3} = .5\\freq = 2\\SR = .125$

> the following is based on top 5 accuracy of Imagenet

The best performing $SR$ was at $.75$, further induced plateaus. Model size increases to 19 MB from the baseline of 4.8MB with $SR = .125$, with a 5.7% increase in accuracy.

The best performing $pcr_{3 \times 3}$ was set to be $.5$, plateauing after at 85.3% accuracy (used an $SR = .5$)


### CNN MACROARCHITECTURE DESIGN SPACE EXPLORATION

Variants, including Residual Connections, ultimately enable better information propagation in the forward pass, as the squeeze layers have a limited set of outputs given $SR = .125$, and the backward pass for better gradient flow (see [here](vxnuaj.com/blog/residuals)).

Empirically, Simple Bypass connections enabled better accuracy than Complex Bypass.

**Simple Bypass**

Residual Connections at Fire Modules $3, 5, 7, 9$

The residual connection is simply the $I$ transformation.

Limitation is that for every Residual Connection, the number of input and output channels have to be the same, so they go with the Complex Bypass for experimentation

**Complex ByPass**

Residual Connections at every Fire Module, where those that aren't previous *simply bypass*, as residual connections with $1 \times 1$ convolutions to reduce $H \times W$ and $C$.

This results in added model size (2.9MB added)

