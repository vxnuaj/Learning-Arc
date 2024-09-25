# Loss Functions

> Learning Resource: [Understanding Deep Learning](https://udlbook.com/) Chapter 3.X

### 5.1

You want to maximize the likelihood / probability output of the model, $f(x, \phi)$, that given an $x$ we get it's correpsonding label $y$. ($\phi$ are the model parameters).

A model can be easily trained to maximize this likelihood by training it to predict the parameters for a probability distribution $\theta$, where $\theta = \set{\mu, \sigma^2}$ in the case of a univariate (distribution for a single number) normal distribution. It's goal would be to find the proper $\theta$ that maximizes the probability of the correct label for $x$, which is $y$, occuring for the given $x$.

The **maximum likelihood criterion**, then computes the total combined probability of all $P(y|x)$, it's goal to maximize it's value. A maximal value of $1$ would indicate that our model is able to accurately map the input $x$ to it's corresponding label $y$.

$\hat{\phi} = \argmax{\prod_i^I [P(y_i | x_i)]}$

This is under the assumptions that the data is independent and identically distaributed ($i. i. d$).

Ideally, we maximize this value to $1$.

But using this as a metric has downfalls as multiplying $n$ probabilities together, decreases the magnitude of the final output to a smaller value as $n$ increases. Ultimately, the result might be very small.

Then instead we can compute the log-liikelihood as a solution:

$\hat{\phi} = argmax[ \sum_i^I[log(P(y_i | x_i))]]$

$\prod$ was turnt to $\sum$ as the logarithm of a given values, $a\cdot b$ turns into $log(a) + log(b)$ when we apply a logarithm as $log(a \cdot b)$.

Of course we could always compute the log-likelihood as:

$\hat{\phi} = argmax[log(\prod_i^I[P(y_i | x_i)])]$

which is equivalent.

Computing the log-likelihood as a criterion to maximize is near equivalent to computing the regular likelihood, as log-likelihood is monotonically increasing within the range that the likelihood increases. 

Same goes for decreasing.

Then if we improve the likelihood (maximize) we also do so the the log-likelihood.

Then we want to minimize the negative log likelihoood as convention, to optimization algorithms.

$\mathcal{L}(\phi) = \hat{\phi} = argmin[ - \sum_i^I[log(P(y_i | x_i))]]$

### 5.2 and 5.3

Computing the loss function for a given model can then be seen as constructing a probability distribution using the neural network based on it's output parameters.

For instance, if we chose the normal distribution,

$f(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$

we'd train a neural network to minimize the loss function,

$\mathcal{L}(\mu, \sigma | x) = argmin[-\sum_i^I[log( \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)))]]$

> well look at this giant mess...

by approximating the best $\sigma$ and $\mu$.

Essentially, the model wants to find the best parameters that minimize the negative log likelihood or equivalently maximize the log likelihood.

In regards to standard neural networks, where most use the softmax function, 

$\hat{\phi} = \frac{e^z}{\sum e^z}$

we want to find the best $z$ that ends up minimizing the negative log likelihood.

Then, you can consider neural networks as maximizing the probability of a given $y$ with the input $x$ through it's parameters, $\phi$.