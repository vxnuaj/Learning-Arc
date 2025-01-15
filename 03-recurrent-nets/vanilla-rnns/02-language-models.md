## Language Models

Assuming that the tokens in a text sequence of length $T$ are $x_1, x_2, \dots, x_T$, the goal of language models are to estimate the joint probabilty, $P(x_1, x_2, \dots, x_T)$


```math
P(x_1, x_2, \dots, x_T) = P(x_1) P(x_2 \mid x_1) P(x_3 \mid x_1, x_2) \dots P(x_T \mid x_1, x_2, \dots, x_{T-1})
```

### Evaluating a Language Model

The cross entropy loss for a language model is given as:

```math

\mathcal{L} = -\sum_{t = 1}^T \hat{p} \cdot \log P(x_t | x_1, \dots, x_{t-1})

```

essentially being a sum of all possible losses 

Typically, in language modelling we use perplexity, defined as:


```math

\mathcal{PPLX} = \exp(\mathcal{L})

```

where the base of the exponential is equivalnet to the base of the logarithm used to compute $\mathcal{L}$.

It's simply an exponential of the loss function.

You want $\mathcal{PPLX}$ to to be computed using the same base in the exponential that was used to compute the logarithm in $\mathcal{L}$, to make it an interpretable and reliable metric over different models, given that raising a logarithm with base $b$ returns the value input to the logarithm itself, $u^{log_u(p)} = p$.

Though given a negative logarithm (cross entropy loss.):

```math

\mathcal{L} = - log_b(p), \hspace{1mm} p \leq 1
\\[3mm]
\mathcal{PPLX} = b^{\mathcal{L}}  = \frac{1}{p}

```

The $\mathcal{PPLX}$ can be interpreted as the fraction, $\frac{1}{p}$ where $p$ is the output of the softmax $\sigma$.

The optimal perplexity value is $1$, meaning the model is completely certain about it's prediction. A higher perplexity, leads to a more unconfident model.

```math

\mathcal{PPLX} \in [1, \infty]

```