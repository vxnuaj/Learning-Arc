## Regularization

### L2 Regularization

L2 regularization punishes the sum of squares of the parameter values, or the squared norm of the weight matrix.

Rather than minimizing the loss as regular, we compute instead:

$\hat{\phi} = argmin_{\phi}[\sum_{i = 1}^I l_i[x_i, y_i] + \lambda \sum_j \phi^2_j]$

with the added $\lambda \sum_j \phi_j^2$ as the term that increases the loss, dependent on the magnitude of params, $\phi$.

Larger values of $\phi$ typically indicate overfitting, as the output varies more when certain $\phi$ are activated and used in the weighted sum, while others are not.

> Remember that high variance in the test set of the neural network indicates that the model is overfitting to the training set.

Adding the penalty term to the loss function, and then during backpropagation as $2\lambda\phi$, increases the magnitude of the gradient and in the update step, as we're subtracting the $\alpha g_t$ from $\phi$, we force $\phi$ to become smaller given that we're adding an extra penalty to the gradient, $g_t$, purely based on the magnitude of the weights.

Thereby, as weights are larger, we perform larger weight updates to keep them smaller.

> *Typically, this is known as weight decay but there's a large difference. Weight decay adds the penalty directly to the weight update as:*

```math

\phi = \phi - \alpha g_t - \lambda \sum_j \phi^t

```

> *While for SGD this might not matter as much, in different varations of optimziers like Adam, RMSProp, or Momentum, this can become fallible as we compute first / second moments based on the gradients. This is why AdamW came along to propose that we use weight decay purely instead of L2 Regularization for other optimizers besides vanilla SGD*

### L1 Regularization

Mechanistically works the same as L2, but instead takes the absolute value of the magnitude of weights, rather than squaring it.

$\hat{\phi} = argmin_{\phi}[\sum_{i = 1}^I l_i[x_i, y_i] + \lambda|\sum_j \phi_j|]$

> *is the gradient of l1 regularization the signed weights or the signed magnitude?*