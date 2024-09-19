<img src = 'img/convolutions.png'>

# Going Deeper with Convolutions

Convoluting $\eta$-dimensional tensors over abstract manifolds.

This is based on my learning arc, which you can check out [here](https://vxnuaj.notion.site/Learning-Arc-1025f1989c4f80539f65f688c2c0a251) or [down below](#learning-arc)!

PLEASE - if you see any slop, my errata, please raise an issue or create a pull request! 

Questions? Message me on [X (formerly twitter) @ vxnuaj](https://x.com/vxnuaj)!

1. [Learning Arc](#learning-arc)
2. [Index](#index)

## Learning Arc

*Last updated 9/09/2024*

> *I wrote this on my own, with the help of resources + GPT. Take it all with a grain of salt.*

<details><summary> Resources </summary>

https://udlbook.github.io/udlbook/

https://www.deeplearningbook.org/

https://cs231n.stanford.edu/

https://atcold.github.io/NYU-DLSP20/

- Extra
    - http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/
    - https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks
    - https://srdas.github.io/DLBook/ImprovingModelGeneralization.html
    - https://colah.github.io/

</details>

### Recap on Feed Forward Neural Networks

> *If you know these well, skip.*
> 
<details><summary>Intro to Neural Networks</summary>
   
  - [ ] NYU Week 1
      - [ ]  https://www.youtube.com/watch?v=0bMe_vCZo30&list=PLLHTzKZzVU9eaEyErdV26ikyolxOsz6mq&index=2&t=4939s
      - [ ]  https://ebetica.github.io/pytorch-Deep-Learning/en/week01/01-1/
      - [ ]  https://ebetica.github.io/pytorch-Deep-Learning/en/week01/01-2/
  - [ ]  3.X (Shallow Neural Networks) of https://udlbook.github.io/udlbook/
  - [ ]  4.X (Deep Neural Networks) of https://udlbook.github.io/udlbook/
  - [ ]  5.X (Loss Functions) of https://udlbook.github.io/udlbook/
  - [ ]  Review K-Fold Cross Validation
  - [ ] NYU Week 2 P1
      - [ ]  https://www.youtube.com/watch?v=d9vdh3b787Y&list=PLLHTzKZzVU9eaEyErdV26ikyolxOsz6mq&index=3
      - [ ]  https://ebetica.github.io/pytorch-Deep-Learning/en/week02/02-1/
      - [ ]  https://ebetica.github.io/pytorch-Deep-Learning/en/week02/02-2/
  - [ ]  Build a Vanilla Neural Network on Fashion-MNIST using Jax, using K-Fold Cross Validation
  - [ ]  6.X (Deep Feedforward Networks) of https://www.deeplearningbook.org/
  - [ ]  Build a Vanilla Neural Network on Fashion-MNIST using PyTorch, using K-Fold Cross Validation.
</details>

<details><summary>Optimization</summary>

> **We will be reviewing Optimizers.** As each is learnt, each will be implemented in Jax and PyTorch on Fashion MNIST.

- [ ]  https://www.youtube.com/watch?v=wEoyxE0GP2M&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=6
- [ ]  https://www.youtube.com/watch?v=_JB0AO7QxSA&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=7
- [ ] 6.X (Fitting models) of https://udlbook.github.io/udlbook/
    1. Read the Papers (or equivalent technical resource) and attempt to understand.
    2. Implement Optimizers within a Neural Network using Jax & K-Fold Cross Validation as they are mentioned
        (rebuild a Neural Network, then use it for the rest of the Optimizers as a base script)
    3. Implement Optimizers in PyTorch with K-Fold Cross Validation
- [ ] Batch Normalization of https://udlbook.github.io/udlbook/
    - [ ]  Read Paper, attempt to understand
    - [ ]  Implement in Jax with K-Fold Cross Validation & Adam Optimizer
    - [ ]  Implement in PyTorch with K-Fold Cross Validation Adam Optimizer
- [ ] Layer Normalization
    - [ ]  Read Paper, attempt to understand
    - [ ]  Implement in Jax with K-Fold Cross Validation & Adam Optimizer
    - [ ]  Implement in PyTorch with K-Fold Cross Validation & Adam Optimizer
- [ ] Group Normalization
    - [ ]  Read Paper, attempt to understand
    - [ ]  Implement in Jax with K-Fold Cross Validation & Adam Optimizer
    - [ ]  Implement in PyTorch with K-Fold Cross Validation & Adam Optimizer
- [ ] 7.X (Gradients and Initialization) of https://udlbook.github.io/udlbook/
    - [ ]  Cover Gradient Clipping
- [ ]  8.X (Measuring Performance) of https://udlbook.github.io/udlbook/
- [ ] 8.X of https://www.deeplearningbook.org/
    1. Read the Papers for the remaining optimizers and attempt to understand
    2. Implement the remaining Optimizers (use the base script) with K-Fold Cross Validation
    3. Implement remaining Optimizers in PyTorch with K-Fold Cross Validation
- [ ] If not learnt, cover these
    > Cover the Theory / Rationale and Implementation
    - [ ] Nesterov Momentum
        - [ ]  Theory / Rationale
        - Implementation in Toy Neural Networks
            - [ ]  Jax.numpy
            - [ ]  Pytorch
    - [ ] Adagrad
        - [ ]  Theory / Rationale
        - Implementation in Toy Neural Networks
            - [ ]  Jax.numpy
            - [ ]  Pytorch
    - [ ] AdamW
        - [ ]  Theory / Rationale
        - [ ] Implementation in Toy Neural Networks
            - [ ]  Jax.numpy
            - [ ]  Pytorch
    - [ ] Nadam
        - [ ]  Theory / Rationale
        - [ ] Implementation in Toy Neural Networks
            - [ ]  Jax.numpy
            - [ ]  Pytorch
</details>

<details><summary>Regularization</summary>

> **We will be reviewing Regularization.** Each will be implemented in Jax + PyTorch on Fashion MNIST, with the Adam Optimizer.

- [ ]  9.X (Regularization) of https://udlbook.github.io/udlbook/
    - [ ]  Read papers when available, if not watch regularization L10 https://sebastianraschka.com/blog/2021/dl-course.html#l10-regularization-to-avoid-overfitting
    - [ ]  Implement mentioned methods within a Neural Network using Jax, alongside Adam Optimizer with K-Fold Cross Validation
        - [ ]  Rebuild the Neural Network
        - [ ]  Implement Regularization
    - [ ]  Implement Regularization in PyTorch with K-Fold Cross Validation
- [ ]  7.X (Regularization for Deep Learning) of https://www.deeplearningbook.org/
    - [ ]  Read Remaining Papers if needed / equivalent theoretical resources
    - [ ]  Implement remaining if needed in Jax & PyTorch with K-Fold Cross Validation

</details>

### **Convolutional Neural Networks**

- [ ] [Video Series: CNN Basics](https://www.youtube.com/watch?v=FW5gFiJb-ig&list=PLLHTzKZzVU9eaEyErdV26ikyolxOsz6mq&index=5)
- [ ] **OPTIONAL**. Extra Insights: [Video Series](https://www.youtube.com/watch?v=bNb2fEVKeEo&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=5)

<details><summary>10.X (Convolutional networks) of https://udlbook.github.io/udlbook/</summary>

- [ ] Read the chapters, do the problems, and go through the notebooks provided.
- [ ] Read Relevant Cited Papers when applicable.
- [ ] Implement every individual important CNN concept introduced, from scratch in Jax and PyTorch

    > **NOTE**: These are to be done separate from the notebook examples. If there is a corresponding notebook example in the book, do that one first and then do the below. If you need to get deeper to understand, via other resources, **do it. Depth matters**

    - **10.2**
        - 10.2.1 | 1D Convolution with Kernel Size
            - [ ] jax.numpy
            - [ ] PyTorch
        - 10.2.2 | 1D Convolution with Kernel Size + Padding
            - [ ] jax.numpy
            - [ ] PyTorch
        - 10.2.3 | 1D Convolution with Kernel Size + Padding + Stride
            - [ ] jax.numpy
            - [ ] PyTorch
        - 10.2.4 | 1D Convolution with Kernel Size + Padding + Stride + Dilation
            - [ ] jax.numpy
            - [ ] PyTorch
        - 10.2.5 | 1D Convolution with Kernel Size + Padding + Stride + Dilation + Multiple Channels
            - [ ] jax.numpy
            - [ ] PyTorch
        - 10.2.6 | Single CNN 1D Convolutional Layer with Kernel Size + Padding + Stride + Dilation + Multiple Channels
            - [ ] jax.numpy
            - [ ] PyTorch
        - 10.2.7 | MNIST-1D Forward Pass
            - [ ] jax.numpy
            - [ ] PyTorch
    - **10.3**
        - 10.3 | 2D Convolution with Kernel Size + Padding + Stride + Dilation + Multiple Channels
            - [ ] jax.numpy
            - [ ] PyTorch
    - **10.4**
        - 10.4.1 | Downsampling Max Pooling Layer
            - [ ] jax.numpy
            - [ ] PyTorch
        - 10.4.1 | Downsampling Mean Pooling Layer
            - [ ] jax.numpy
            - [ ] PyTorch
        - 10.4.2 | Nearest Neighbor Upsampling
            - [ ] jax.numpy
            - [ ] PyTorch
        - 10.4.2 | Max Unpooling
            - [ ] jax.numpy
            - [ ] PyTorch
        - 10.4.2 | Bilinear Interpolation
            - [ ] jax.numpy
            - [ ] PyTorch
        - 10.4.2 | Transposed Convolution
            - [ ] jax.numpy
            - [ ] PyTorch

</details>

<details><summary>Paper Implementations</summary>

- Toy CNNs
    > Implement from scratch on one, then use as a blueprint for the variants below
    - [ ] Implement a Toy CNN on fashion-mnist in jax.numpy from scratch
    - [ ] Implement a Toy CNN on fashion-mnist in PyTorch from scratch
    - Spatial Dropout
        - [ ] jax.numpy
        - [ ] PyTorch
    - Spatial BatchNorm
        - [ ] jax.numpy
        - [ ] PyTorch
    - Spatial LayerNorm
        - [ ] jax.numpy
        - [ ] PyTorch
    - GroupNorm
        - [ ] jax.numpy
        - [ ] PyTorch
    - Momentum
        - [ ] jax.numpy
        - [ ] PyTorch
    - RMSprop
        - [ ] jax.numpy
        - [ ] PyTorch
    - Adam
        - [ ] jax.numpy
        - [ ] PyTorch
    - AdamW
        - [ ] jax.numpy
        - [ ] PyTorch
- [LeNet-1](https://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf), [LeNet-4](https://yann.lecun.com/exdb/publis/pdf/lecun-90c.pdf), [LeNet-5](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
    > For Each, learn from / do the following
    - [ ] Technical Video Explanation
    - [ ] Read & Understand Paper
    - [ ] Implementation with KFold Validation
- [AlexNet](https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf), [Squeeze Net](https://arxiv.org/pdf/1602.07360)
    > For Each
    - [ ] Technical Video Explanation
    - [ ] Read & Understand Paper
    - [ ] Implementation with KFold Validation
- [ResNet](https://arxiv.org/abs/1512.03385), [ResNet-V2](https://arxiv.org/abs/1603.05027), [ResNeXt](https://arxiv.org/abs/1611.05431v2)
    > For Each
    - [ ] Technical Video Explanation
    - [ ] Read & Understand Paper
    - [ ] Implementation with KFold Validation
- [Inception V1](https://arxiv.org/abs/1409.4842), [V2](https://arxiv.org/abs/1502.03167v3), [V3](https://arxiv.org/abs/1512.00567v3), [V4](https://arxiv.org/abs/1602.07261)
    > For Each
    - [ ] Technical Video Explanation
    - [ ] Read & Understand Paper
    - [ ] Implementation with KFold Validation
- [ ] [Video Series: Advanced CNNs](https://www.youtube.com/watch?v=DAOcjicFr1Y&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=9)

</details>

<details><summary>9.X (Convolutional networks) of https://www.deeplearningbook.org/</summary>

- Implement these concepts, in jax.numpy and pytorch. Add others if you see that you need for understanding.
    > **NOTE** Some will be reimplemented for understanding.
    - **9.1**
        - 2D Convolution with Kernel Size + Padding + Stride + Dilation + Multiple Channels
            - [ ] jax.numpy
            - [ ] PyTorch
    - **9.3**
        - Max Pooling
            - [ ] jax.numpy
            - [ ] PyTorch
        - Average Pooling
            - [ ] jax.numpy
            - [ ] PyTorch
- Read Relevant Cited Papers (if not done prior)
- If not learned, cover these
    > Cover the Theory / Rationale and Implementation
    - Advanced Convolutions
        - [ ] Depthwise Separable Convolutions
    - Advanced Pooling
        - [ ] Global Max Pooling
        - [ ] Global Average Pooling
    - Advanced Activation Function Variants
        - [ ] Parametric ReLU
        - [ ] ELU
        - [ ] SELU
        - [ ] Swish
        - [ ] GELU

</details>

<details><summary>Implementations V2</summary>

- [VGGNet](https://arxiv.org/abs/1409.1556)
    > For Each
    - [ ] Technical Video Explanation
    - [ ] Read & Understand Paper
    - [ ] Implementation
- [DenseNet](https://arxiv.org/abs/1608.06993)
    - [ ] Technical Video Explanation
    - [ ] Read & Understand Paper
    - [ ] Implementation
- [MobileNet V1](https://arxiv.org/abs/1704.04861), [MobileNet V2](https://arxiv.org/abs/1801.04381v4)
    > For Each
    - [ ] Technical Video Explanation
    - [ ] Read & Understand Paper
    - [ ] Implementation
- R-CNN Series or [Xception](https://arxiv.org/abs/1610.02357v3)
    > For Each
    - [ ] Video Technical Explanation
    - [ ] Understand Paper

</details>

## Index

1. [Intro / Recap to Neural Networks](intro-to-nnet)
   1. [NOTES | NYU Lecture: History, motivation, and evolution of Deep Learning](intro-to-nnet/01-nyu-1.md)
   2. [NOTES | Chapter 3 of Understanding Deep Learning](intro-to-nnet/02-Chap3UDL.md)
   3. [NOTES | Chapter 4 of Understanding Deep Learning](intro-to-nnet/03-Chap4UDL.md)
   4. [NOTES | Chapter 5 of Understanding Deep Learning](intro-to-nnet/04-Chap5UDL.md)
   5. [IMPLEMENTATION | Neural Net from Scratch in Jax](intro-to-nnet/06-nn-jax.py)
   6. [IMPLEMENTATION | Neural Net from Scratch in NumPy](intro-to-nnet/06-nn-np.py)