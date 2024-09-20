<img src = https://i.pinimg.com/736x/68/85/82/68858225932ab1636969521afc30b60b.jpg width = 1000>

# Deep Forward Networks

> Learning Resources: [Deep Learning Book](https://www.deeplearningbook.org/) Chapter 6.X

The goal of a feedforward network, a classifier for instance, is to approximate a function $f^*$, where the classifier $y = f^*(x; \theta)$ maps the given $x$ to the category $y$.

The model learns the appropriate function parameters, $\theta$, to give us the $y$ from the $x$, by approximaing the function $f^*$.

They are feedforward, as information flows purely forward throug, to the putput $y$. There are no feedback connections (RNNs) in a feed forard model.

They are called networks because they are composed of multiple smalelr functions, $f^i$ (a given layer). The larger the total layers, $I$, the deeper the model is.

> *Review...*

Dimensionality of each layer (num of neurons / hidden units) determines it's *width*. 

Instead of thinking of each layer like a vector-to-vector function, get more specific and think of each layer as set of hidden units that represent vector-to-scalar functions.

Best to think of neural nets as funciton approximation machines taht achieve statistical generalization, which are loosely based on the brain's functions. They aren't models of the brain itself (ofc).

Non-linearity is what lets neural networks beat linear models. 

Without an activation function,$\phi$, there is no use for a deep model, might as well train a shallow network or a perceptron (given that your output vectors after each layer will stay on the same subspace, and are linearly dependent to each other, because of a lack of a non-linearity)

Previously, you could've created non-linearities ($\phi$) via:

1. $\phi$ via the kernel tricks
2. Manually engineer $\phi$ (oh lord)

but now we have deep learning. It captures the benefits of those two approaches, where $\phi$ can be approximated in a generic manner, unlike prior where a specific $\phi$ might've differed for each domain, and where human practitioners can encode generic knowledge in the $\phi$ by only defining a general function or in this case, a deep model.

### 6.2.x

Neural networks are trained typically via maximum like-likelihood, which is analagous to maximum log-likelihood, given that they are both monotonically increasing over the same domain, albeit log-likelihood is on the log scale.

The cost function then easily becomes the negative log-likelihood where we want to train a model to minimize the negative log likelihood (NLL).

$NLL  = Cross Entropy$ betweeen training data and model distribution

**Maximum Likelihood Criterion:** $argmax[\prod_i^I [p(y_i | f(x_i, \theta))]]$

**Maximum Log-Likehood Criterion:** $argmax[log\prod_i^I[p(y_i | f(x_i, \theta))]] = argmax[\sum_i^I[log(p(y_i | f(x_i, \theta))]]]$

**Minimizing Log-Likelihood Criterion**

$J(\theta) = -\mathbb{E}_{(x,y) \sim \hat{p}_{\text{data}}} \left[ \log p_{\text{model}}(y | f(x, \theta)) \right]$

$argmin(J(\theta))$

The entire function changes from model to model depending on the specific $p$ thast the model uses.

For multiclass classification, it's typically softmax as $\frac{e^z}{\sum{e^z}}$, thereby the loss criterion becomes:

$J(\theta) = -\mathbb{E}_{(x,y) \sim \hat{p}_{\text{data}}} \left[ \log \frac{e^z}{\sum{e^z}} \right]$

Depending on the model, there may be formulations of cost functions that yield constants, depenidng on the form of $p$ for the given model.

When taking the gradient of these loss functions, with added constants, we can effectively remove them as when we take the gradient of the loss w.r.t to $\theta$, the constants become $0$. 

Thereby they have no effect on minimizing the loss via updates to $\theta$ in convex based optimization.

**ESSENTIALLY, MAXIMUM LIKELIHOOD SERVES AS A BLUEPRINT FOR DEFINING COST FUNCTIONS. MODEL YOUR LOSS FUNCTION, BASED OFF A PROBABILITY OUTPUT, WHICH IS THEN PASED ON PRIORS OF $\theta$**

But of course, the gradient of the cost function **must** be large and predictable enough to serve as a guide for the neural network. 

Cost functions that become **flat** are bad. We don't want **flat gradients** 

> is why sigmoid and tanh are problems in deep networks as well, vanishing gradients.

The issue with many output units in a neural network, are that their activation, $\phi$, can include many $exp$ functions.

Those functions can have flat gradients when it's input is negative (i.e., when the input to softmax is $-10$), and therefore the $log$ function in $NLL$ allows for us to rid of the saturation and introduce a more steeper gradient, in the right direction.

   $$ p(y_i | x) = \frac{e^{z_i}}{\sum_{j} e^{z_j}} $$

   $$ z = [-10, 0, 1] $$

   $$ p(y_1 | x) = \frac{e^{-10}}{e^{-10} + e^{0} + e^{1}} = \frac{0.0000454}{0.0000454 + 1 + 2.718} \approx \frac{0.0000454}{3.718} \approx 0.0000122 $$

It's at this stage, where the log likelihood transofrms the small $.00000122$ into a $10.4$, then yielding us a larger value. 

   $$ \text{NLL} = -\log(p(y_1 | x)) \approx -\log(0.0000122) \approx 10.4 $$

The gradient, considering the surrounding values, is steeper when applying the $\text{NLL}$

   $$ \frac{\partial \text{NLL}}{\partial z_1} = -\frac{1}{p(y_1 | x)} \cdot \frac{\partial p(y_1 | x)}{\partial z_1} $$

Left off at 6.2.2.3