# More on Receptive Fields

The receptive field of the $lth$ layer, is the dimensionality of the extracted region of the original input to the model $f$, given by all kernels up to the $lth$ layer, ($\mathcal{K^l}, \forall l \in \text{range}(L_{curr})$), that a single feature in the output feature map for layer $l$ "sees" or is extracted from.

The dimensionality of the receptive field for a model increases as the model **depth** increases.

The deeper a model is, meaning the more convolutional and pooling layers it has, the more information it's able to extract and then summarize, as a single feature in a smaller output feature map.

**As the model gets deeper, the size of the receptive field increases with respect to a given single feature in the output feature map**

<br/>
<div align = 'center'>
<img width = 500 src = 'https://miro.medium.com/v2/resize:fit:1200/1*k97NVvlMkRXau-uItlq5Gw.png'>
</img>
</div>
<br>

Note how for the 2nd layer, we have a receptive field, of size $3 \times 3$, from $l = 1$. By the third layer, given that we're extracting features from $l = 2$, with a region size of $3 \times 3$, and that each feature in $l = 2$ has a receptive field of $3 \times 3$ as well, the receptive field for $l = 3$ amounts to being $5 \times 5$.

Smaller receptive fields more strongly invoke the principle of locality, where given an input image $X$, we hypothesize that the important feature needed to classify an object, $\mathcal{O}$, into it's rightful class, can be done by purely extracting a smaller set of **local** feature from the original input (given the smaller receptive field).

Larger receptive fields introduce a more global context to the model, allowing neurons to see more of the input image. 

Having a larger receptive field, and then downsampling feature maps through convolutions and pooling layers, helps a model extract the more useful information or more important features while summarzing the original features (intelligence is compression, in terms of intelligence to in distribution data, but not generalization to out of distribution data).

You can increase the size of your receptive field by either **increasing** the size of $\mathcal{K}$ **earlier**, at the input, or simply making your model **deeper**.

As noted above, making your model deeper, despite having a smaller **kernel**, can still allow you to get global context of the original input (see image), again **despite having a smaller kernel** in the input. 

You're also able to extract more relevant hierarchical features given that we have a larger model capacity through deeper layers.

Having a larger kernel, right at the input, offsets your need for deeper layers to get global context via a larger receptive field, with the diminishing return that the extracted features may not be as meaningful as they weren't hierarchically extracted.

The more layers we have, the more specific, nuanced, and important features we're able to extract from the original input, such that we're able to more accurately classify samples, alongside a larger receptive field as well. The only risk to this is overfitting and increased computational compelxity.

