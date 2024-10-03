# Shallow Neural Networks

> Learning Resource: [Understanding Deep Learning](https://udlbook.com/) Chapter 3.X

### 3.1 Neural Network Example

Shallow Neural Networks (2-layer) are a function $y = f[x, \phi]$, that maps inputs $x$ to $y$ with trainable parameters $\theta$.

$f(x, \phi) = \phi_0 + \phi_1a(\theta_{10} + \theta_{11}x_1 + ... + \theta_{1m}x_i) + ... + \phi_ka(\theta_{n0} +... +\theta_{nm}x_i)$

where:

- $x$ are the inputs
- $\theta$ are the weights of the input layer
- $a$ is the activation function
- $\phi$ are the weights of the output layer.<br>

Typically $a$ is defined as $ReLU(z)$, which is $max(0, z)$

To effectively map $x$ to $y$ via $f$ we can minimize a given loss function $L$, say the mean squared error, by optimizing for the optimal $\phi$, $\hat{\phi}$

### 3.2 Univ. Approx. Theorem

The number of hidden units, $h$, of a neural network represents it's capacity, where the network capacity is the amount of complex functions it can approximate.

Hypothetically, a neural network with a single hidden layer can approximate any function given an infinute number of hidden units

### 3.3 Multivariate Inputs & Outputs.


For multivariate outputs (multiple output neurons), the 'joints' of a final piecewise functions (of each neuron), of the outputs, depend on where the initial linear functions $\theta_{n0} + ...+ \theta_{nm}x_i$ are cliipped by the $ReLU()$ activation.

But each will have their 'joints' at the same location, given that each output neuron is connected to the same hidden units in the previous layer.

So the final piecewise function has joints at the same positions, given that the neurons in the output layer share the same hidden layer inputs. The prior hidden layer inputs are all clipped by $ReLU$ at the same position, where the input to $ReLU$ turns to $x ≤ 0$.

But each output neuron has a different final piecewise function, despite sharing the same inputs and 'joints'. This is as each $i$th output neuron has a different set of parameters (weights and a bias) for each $n$ unique connection, ultimately altering the function per output neuron.

In regards to multivariate inputs, the combination of piecewise $ReLU$ activations, creates a surface of convex polygonal regions, given by the intersecting hyperplanes of each hidden unit in the $\mathbb{R}^n$ space.

Think of the construction of hyperplanes as such:

For each $ith$ hidden unit, there is an equation $\theta_ix + b_i$ that yields the scalar output for the given $ith$ hidden unit. 

Given the $ReLU$ function as $max(0, x)$, it clips the output of the linear combination to $0$ where $x < 0$. The values of $x$ that yield $0$, when the output of the linear combination is applied onto $ReLU$ represent the position of the diviing hyperplanes.

Each intersection of the hyperplanes creates a seperate convex polygonal region. If we had 2 intersectiong hyperplanes (2 hidden units) in an $\mathbb{R}^2$ space we'd be dividing the region into 4 seperate convex polygons.

> *a function is convex if a line drawn between any two points on it lies above the minima. inverse is concave*

Another example,

<img src = ../img/poly.png>
<br><br>

where 3 hidden units, $h_i$ divide the 2 dimensional space, $\mathbb{R}^2$ into 7 distinct regions.

As the number of input dimensions (features) grows, the number of convex polygonal regions begins to increaes exponentially. If you imagine each extra hidden unit as an additional hyperplane in the $R^{n}$ space, and we had the same number of hidden units as input dimensions, the number of convex polygonal regions increases by a rate of $2^n$, where $n =$ input dimensions, under ideal conditions where the hyperplanes are optimally placed.

Of course, neural networks often have more hidden units (hence deep) than input dimensions, so the number of convex polygonal regions is typically higher than $2^n$.

It's better approximated by $2^k$ where $k$ is the number of hidden units in the neural network.

> *This phenomen can be known as the curse of dimensionality, the more dimensions you have, the more exponentially complex your problem space becomes.*

The maximum number of regions created by $m$ hyperplanes in an $n$-dimensional space can be defined as a sum of binomial coefficients, $R_n(m) = \sum_{j=0}^{n}\begin{pmatrix}m \\ j\end{pmatrix} = \frac{m!}{j! \cdot (m-j)!}$. 

This serves as the upper bound of possible regions, considering all possible ways those hyperplanes can intersect. It assumes that each hyperplane contributes to the given regions as optimally as possible under the following:

- Hyperplanes are not parallel

- No more than $n$ hyperplanes intersect at the same $n-1$ subspace within $\mathbb{R}^n$ (essentially, does each hyperplane add unique information at every point?)

If we're operating in $\mathbb{R}^3$, no more than $3$ hyperplanes can intersect in the same $2$-dimensional subspace. While 2 hyperplanes themselves can create a line on their own, a 3rd hyperplane still adds new information (hence, independence of the hyperplane) in the 3rd direction of the $\mathbb{R}^3$ space, to make use of the full $3$-dimensional space. Otherwise, the line created by $2$ hyperplanes alone would be merely within an $\mathbb{R}^2$ subspace of $\mathbb{R}^3$.

> *For anyone reading this, please learn linear algebra if you haven't yet.*

**TLDR**:

Each hidden unit creates a hyperplane. The hyperplanes (or hidden units) creates a unique convex polygonal region in the $R^n$ space. The maximum number of possible regions created by $m$ hyperplanes can be $R_n(m) = \sum_{j=0}^{n}\begin{pmatrix}m \\ j\end{pmatrix} = \frac{m!}{j! \cdot (m-j)!}$. This assumes that the hyperplanes are not parallel and not more than $n$ hyperplanes intersect at the same $n-1$ subspace in the $R^n$ space (essentially, does each hyperplane add unique information at every point?)

## Other Notes

### On Activation Functions

ReLU is great, given that it has a derivatie which is always greater than $0$ when the input, $x > 0$ Unfortunately, it's derivative is $0$ for negative inputs. When a neural network has all negative inputs to a given $ReLU$ it's gradient will always be $0$ and therefore we can't update the weights via $w - \eta\frac{∂L}{∂\phi}$ as $\frac{∂L}{∂\phi}$ is always $0$. This is the dying $ReLU$ problem. Solved by it's variants.

A homogenous function is a function, say $f$, where:

$f(tx) = t^kf(x)$

where $k$ is the degree of homogeneity for the given function and $t$ is any scalar multiple.

$ReLU$, has the property of $k \cdot ReLU(x) = ReLU(kx)$, it's degree of homogeneity is $1$, for non negative inputs. 

Sigmoid $\sigma$ and tanh have the vanishing gradient problem, where their gradient is extremely small for large negative or positive inputs.





___