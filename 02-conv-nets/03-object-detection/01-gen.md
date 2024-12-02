### Bounding Boxes

Boxes around objects of interest in a given image, *bounding* the object to the area determined by the bounding box.

### Model Outputs

For a given model $\mathcal{F}$, we can classify an image $X$, to it's rightful class via a softmax output, which directly identifies the class index for the image.

Another option is to have $\mathcal{F}$ output the parameters for the bounding boxes, $b_x$ (x position), $b_y$ (y position), $b_h$ (height), and $b_w$ (width), which then correspond to the placement and spat ial size of the bounding box to be constructed on the image.

> $b_x$ and $b_y$ are the central pixels of the bounding box.

Then we also have $p_c$, which is a binary choice, denoting if there is an object (if one of the class object is in the image).

> $p_c$ is a probability, in the range from $[0, 1]$
> 
> Note that we also have a background class, which is the identiifed class if none of the class objects are in the image.
>
> $p_c = 0$ if the model classifies to the background class (or in practice, near $0$, with background class being a high probability.)

So the model outputs $c_1, c_2, \dots ,c_i$, where $i$ is the total number of classes, as well as all $b_n$, where $n \in [x, y, h, w]$, and finally $p_c$.

```math

\hat{Y} = \begin{bmatrix}p_c \\ b_x \\ b_y \\ b_h \\ b_w \\ c_1 \\ c_2 \\ \vdots \\ c_i\end{bmatrix}

```

During inference, if $p_c = 0$ (rounded sigmoid logit), then we ignore all other components of the output vector.

You can also add $l_{xy}$ values to the output to detect specific postiions of specific landmarks you want to detect in an image.

### Loss Functions.

Object detection models use multiple loss functions, to compute an overarching loss for the model.

You have different loss functions for **localization**, **classification**, and **objectness / confidence loss**.

**Classification**

- Cross Entropy Loss
- Focal Loss

**Localization Loss:**

- MAE
- MSE
- IoU based loss functions

**Objectness / Confidence Loss**

- Binary Cross Entropy
- Hinge Loss

Each output that corresponds to the given task is applied with the specified loss and then the total loss is computed as:

```math

\mathcal{L} = \lambda_1 \mathcal{L}_c + \lambda_2 \mathcal{L}_L + \mathcal{L}_O \lambda_3

```

where $\lambda_i$ is the hyperparameter controlling the influence of each loss on the total loss.

### Sliding Windows.

For a ConvNet to detect if there is an object at a given region, one could slide a window over the image to extract a local patchi and then run that patch through the ConvNet.

This could be done for all positions in the image, to get a prediction for all patches.

Then to get the bounding box, we run different sizes of the extracted window (denoting all possible bounding boxes) until we construct the optimal bounding box.

### Convolutional Implementation of Sliding Windows.

The former tends to be computationally expensive.

So instead we can feed the entire image to a given ConvNet, $\mathcal{F}$, and get the output size to be the total size of the desired extracted windows from the input.

Then each output logit corresponds to the raw probability score of **objectness** for a given window.

<div align = 'center'>
<img src = 'im1.png'></img>
</div>

The raw logit is then passed through a sigmoid activation or another binary activation function, to get a probability estimate.

### Bounding Box Predictions based on Grids

The former doesn't yield accurate bounding boxes as the sliding windows approach only allows for fixed sized bounding boxes, it's likely the bounding box won't be variable despite there possibly being variable sized objects in the image.

Instead what we can do is segment the image as an $n \times n$ grid, where $n$ is chosen based on the sizes of possible regions that contains only $1$ object, typically as a balance if there are mixed sizes.

Then we can run the image through the ConvNet, to get an output of $n \times n \times (5 + c)$, where $c$ is the total number of classes and $5$ corresponds to $[p_c, b_x, b_y, b_h, b_w]$.

Each $i_{out} \in n \times n$, we have a vector of size $5 + c$, containing the prior formulation for an output vector of:

```math

\hat{Y}_g = \begin{bmatrix}p_c \\ b_x \\ b_y \\ b_h \\ b_w \\ c_1 \\ \vdots \\ c_n\end{bmatrix}

```

where $g$ is the index for the current grid portion.

So each grid outputs an **objectness** probability, **localization outputs** for the bounding box, and **class probabilities**.

### Intersection over Union

A metric for localization, akin to accuracy.

For a intersecting regions of hard truth and predicted bounding boxes of size $n \times n$ and a union of hard-truth and predicted bounding boxes of size $m \times m$, the intersection over union is computed as the ratio of the size of $n \times n$ relative to $m \times m$.

```math

\text{IOU} = \frac{n \times n}{m \times m}

```

Typically if $\text{IOU} ≥ 0.5$, then the prediction is denoted as correct. Other values also work.

### Non Max Suppression

A ConvNet with grid size regions smaller than a given object will be prone to detecting multiple bounding boxes for the same object.

Therefore, you can simply select the bounding box with the maximum probability, $p_c$, as the prediction.

### Anchor Boxes

> *Reminder that there is one output $\hat{Y}_i$ for each grid cell, which forms part of the larger $\hat{Y}$ of the model.*

Anchor boxes act as a guide for models to generate predictions for a given grid cell, allowing a pre-specified region to be used as a relative template for generating $b_x, b_y, b_h, b_w$, as they they correspond to the relative ratio of the size of the predicted bounding box to the anchor box.

> *$p_c$* is still generated based on the grid cell independently.

Having multiple anchor boxes, for a single grid cell, will allow you to predict multiple bounding boxes for each grid cell, allowing you to recogize objects that could potentially overlap.

The shape of the specific anchor box will serve as a guide, forcing the ConvNet to generate a prediction that corresponds to the right object.

If you have two overlapping objects, one vertical and another horizontal, a vertically designed anchor box on the vertical object will allow you to create a bounding box for the vertical object rather than the horizontal as it forces the ConvNet to generate parameters for a vertical box, such that the only rational choice (via backprop), is the vertical box for the vertical object.