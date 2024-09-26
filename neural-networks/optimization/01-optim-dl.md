<img src = 'https://losslandscape.com/wp-content/uploads/2019/09/loss-landscape-research-3.jpg'>

# Optimization and Deep Learning

<details><summary>Learning Resources</summary>

- [D2l.ai, Optimization Algorithms](https://d2l.ai/chapter_optimization/index.html)
- [gpt4o, my technical co-founder](chatgpt.com)

</details>

### 12.1.1 Goal of Optimization

While optimization serves as a means to minimize a loss function in deep learning, mathematical optimization and deep learning differ.

Deep Learning minimizes the loss function to the point to where generalization is maximized, while optimization always tries to find the absolute global minima of a given function.

Generalizing doesn't always align with the global minima, therefore we need to find a way to prevent overfitting.

You want to minimize risk, not empirical risk (risk of misprediction on training set).

### 12.1.2 Challenge of Optimization

For a function $f$, if the value of $f$ at $x$ is smaller than other values of $f$ for other $x$, but not the the absolute minima, $f(x)$ is a local minima. 

The function $f$ is not fully convex.

Then the loss function might only minimize for the objective locally, instead of globally. Only a bit of noise can help move the parameter past the local minima, which is an advantage of minibatch gradient descent.

**Saddle** points are another issue. Given $f(x) = x^3$, it has a $0$ gradient when $x = 0$. There will be no weight update at this point and the model will stop training, unless we apply an exponentially weighted average to the gradient calculation (Adam, RMSprop, etc).

Saddle points get worse as we begin operating in higher $n$-dimensional spaces, $\mathbb{R}^n$, where say $n > 2$.

Given a function $f(x, y) = x^2 - y^2$, it has a saddle point at $0, 0$.

The input to this function is a vector in $\mathbb{R}^2$, therefore it's Hessian matrix (see [other notes](#other-notes)), will have $2$ eigenvalues.

When the eigenvalues for the Hessian matrix of the function, are positive where the gradient is $0$, we have a local minima. This is as positive eigenvalues indicate an increasing 1st order partial derivative. But we have a flat, zero-gradient, so therefore we've reached at least a local minima, if not a global minima, a flat point in a convex region.

On the other hand, when the eigenvalues for the Hessian matrix of the function are negative, (meaning the Hessian is negative semi-definite), at a point where we have a $0$ gradient, we've reached at least a local maxima if not a global maxima, indicating a flat point in a concave region.

### 12.1.3 Vanishing Gradients

$Tanh$ is a horrible function for deep models. Plug in a very large or negative $x$ to the function, take the gradient, and see why.

### Other Notes

#### **Eigendecomposition**

Eigendecomposition expresses a square matrix $A$ in terms of it's eigenvalues and eigenvectors. 

> $A$ does not have to be full-rank btw. Just must be square.

$A = V\Lambda V^-1$

where 

- $V$ is a matrix whose columns are eigenvectors of $A$
- $\Lambda$ is a diagonal matrix, containing the eigenvalues of all $A$ on the diagonal

Given $A$, we have an eigenvalue and an eigenvector that satisfies:

$Av = \lambda v$

When we apply a linear combination of the column vectors of $A$ with $v$ as $Av$, we get a multiple of $v$, the multiplier denoted by $\lambda$ which is the eigenvalue and the eigenvector is $v$.

To solve for eigenvectors and eigenvalues:

1. Find the characteristic polynomial, via $det(A - \lambda I)$
2. Solve for each $\lambda$ in the characteristic polynomial
3. Solve for each eigenvector, $v$, by plugging $\lambda$ into $A - \lambda I = 0$

and to finally perform eigendecomposition, $A = V\Lambda V^-1$:

1. Plug in each eigenvalue into the diagonal matrix, $\Lambda$ into the diagonal components.
2. Plug in the corresponding eigenvector to each column vector in $\Lambda$, they must be same indices. Index of eigenvalue column in $\Lambda$ must be the same as index of eigenvector in $V$.

#### **Higher Order Partial Derivatives**

The $n$-order derivatives where $n > 1$, for a multi-variable function $f$. 

You can compute mixed derivatives:

```
     f(x, y)
        /  \
      ∂f_x  ∂f_y -> ∂f_{yx} or ∂f_{yy}
      /  \    
∂f_{xy} ∂f_{xx} 

```

#### **Hessian Matrices**

Given a scalar-valued function, $f$, the Hessian matrix is the construction of the 2nd order partial derivatives of the function $f$.

Denoted as $H(f)$.

Given a function $f(x_1, ..., x_n)$, the structure of $H(f)$ is as:

$H(f) = \begin{pmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2} \end{pmatrix}$

> **NOTATION:**

In $\frac{∂^2f}{∂x_m∂x_n}$, we computed $\frac{∂f}{∂x_n}$ first.

The $∂$ w.r.t **right most** term is comptued first.

> For an n by n matrix, there will be n eigenvalues.

For a given loss function, it is desired that we have a positive semi-definite Hessian matrix, it's eigenvalues being 0, indicating that the function is convex.




**PROPERTIES**

- The Hessian matrix is symmetric, where $\frac{∂^2f}{∂x_m∂x_n} = \frac{∂^2f}{∂x_m∂x_n}$ , following Schwarz's theorem, where if the second order partial derivatives form a continuous functions, the mixed partials are equal to each other.
- If we don't have an $H(f)$, the function is not differentiable, up to the second order., we dont' have guaranteed information about it's concavity (note that second order derivatives determine the change, of the rate of change. Think acceleration in physics).
- The matrix is always symmetric, an $n\times n$ matrix if $f$ if a function of $n$ variables


---