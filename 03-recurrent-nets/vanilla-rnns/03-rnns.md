Markov Models for language modeling involves computing $P$ of a token at a current time step given all other tokens (with constraint of the context size) at previous time steps (in the case of $n$-gram modelling, $n > 1$)

The assumption with this type of model is that we can reliable estimate the probability of a word at $t$ given the history of the previous few words, say $\in [t-1, t-2]$ (trigram).

You can call this the *markov* assumption.

Given a sequence "Hi my name is Jeffrey Dean":

The unigram probability for Jeffrey is simply:

```math

\frac{1}{6}

```

The bigram probability, assuming given "is":

```math

\frac{1}{1}

```

The bigram probability, assuming given "name"

```math

\frac{0}{1} = 0

```

Assuming a larger sequence: "Hi my name is Jeffrey Dean. Hello, my name is Blanca"

The bigram probability of Jeffrey, assuming given "is"

```math

\frac{1}{2}

```

Trigram of Jeffrey given name is:

```math

\frac{1}{2}

```

More generally: 

$P(\text{word}_t | \text{word}_{t-2, \text{word}_{t-1}, \dots}) = \frac{\text{count of input n-gram}}{\text{count of n-1-gram.}}$

The issue comes when considering the size of your $n$-gram, is that for larger $n$, the size of your total probabilities to be stored increases exponentially with the size of your vocabulary,

```math

|V|^n

```

Rather than doing this (computing conditional probabilities for increasing $n$), we can use a model with a hidden state, $h_{t-1}$, which models all the relevasnt information from the previous tokens.

Instead of modelling:

```math

P(x_t | x_{t-1}, x_{t-2}, x_{t-3}, \dots)

```

we can do as:


```math

P(x_t | h_{t-1})

```

as $h_{t-1}$ captures all the relevant information for all previous tokens, and can be constructed as:

```math

h_t= f(x_t, h_{t-1})

```

### Recurrent Neural Networks

- $n$ is the batch size
- $d$ is the dimension of the input embeddings.
- $h$ is the number of hidden units

Assume a minibatch of inputs at time step $t$, denoted as $X_t \in \mathbb{R}^{n \times d}$.

$W_{xh} \in \mathbb{R}^{d \times h}$ is the set of weights for the input $X_t \in \mathbb{R}^{n \times d}$, to get $H_t \in \mathbb{R}^{n \times h}$.

```math

H_t = \phi(X_tW_{xh} + b_h)

```

where $\phi$ is the activation function.

In an RNN, we save the hidden layer outputs, $H_{t-1}$ from the previous time steps and introduce a set of parameters, $W_{hh} \in \mathbb{R}^{h \times h}$, which describes how we use the hidden layer outputs at previous time steps to incorporate within the current hidden layer outputs at $t$.

```math

H_t = \phi(X_tW_{xh} + H_{t-1}W_{hh} + b_h)

```

Seeing the addition operand to combine the hidden state at $t - n$ and the output to the hidden layer at $t$ had me wondering, why not apply an element-wise multiplication instead?

Let's derive backpropagation, given $\phi(f(x) + y)$ where $y$ is the hidden state and $f(x)$ is the hidden layer at current time $t$.

For addition:

```math

\frac{\partial }{\partial x}\phi(f(x) + y) = \phi'(f(x) + y) * f'(x)

```

For multiplication (by the product rule):

```math

\frac{\partial }{\partial x}\phi(f(x) * y) = \phi'(f(x) * y) * f'(x) * y

```

It's clear that backpropagating through a multiplication operation requires more computation, depending on the operation $f'(x) * y$ while addition only requires $f'(x)$.

More importantly,  of $y >> 1$ or $y << 1$, the deeper the RNN goes, the more prominent the problem of exploding or vanishing gradients can become.

Moving on,

Note that for $H$ at previous time steps, the parameters are $\in \mathbb{h \times h}$ rather than $\mathbb{R}^{d \times h}$, as to match the element-wise addition with $X_tW_{xh}$, the matmul of $H_{t-1}W_{hh}$ must result in the same size, $\mathbb{R}^{n \times h}$.

Then we have the output layer as:

```math

\hat{Y}_t = \text{softmax}(H_tW_{hq} + b_q)

```

Unlike Markov models, where as $n$ increases ( or equivalently, as the number of time-steps increases ) the number of probability-values increases, the number of hidden states, $H$ ( or $h$ in prior notation ), doesn't increase as $n$ does so.

This is the key distinction between why RNNs work well over Markov Models, especially as the dimensionality of $W_{hh}$ increases (scaling laws). We're able to summarize prior hidden states within a single matrix, $W_{hh}$, rather than having a separate value for each.

Note that $\hat{Y}_t$ corresponds to the output for a current sequence $S$, purely at time $t$.

For intuition, consider the tokenized sequence (not embedded), 

```math

X_0: [19, 7, 4, 26, 19]\\[3mm]
Y_0: [ 7, 4, 26, 19, 8]

```

The output $\hat{Y}_0$ would depend on the input $x_0 = 19$, ideally being equivalent to $y_0 = 7$ 

Then for $\hat{Y}_1$, the output would depend on $x_1$ as the input and the prior hidden state of $x_0$, to ideally get $y_1 = 4$.

Then for $\hat{Y}_2$, the output would depend on $x_2$ as the input and the prior hidden state of $x_0$ and $x_1, to ideally get $y_1 = 26$.

Then during backpropagation, we backpropagate over all outputs for the current sequence, $\hat{Y}_t, t\in [0, 4]$.

Of course, in practice the input sequences $X_0$ would be constructed as embeddings $\in \mathbb{R}^{n \times d}$, where $d$ is the dimensionality of the embedding space and $n$ is the sequence length.

### Backpropagation Through Time

We can simply compute the loss as:

```math

L = \frac{1}{T} \sum_{t = 1}^T l(A_t)

```

essentially the loss for all outputs averaged over all time steps.

Remember,

- Input $X_t^{(1)} \in \mathbb{R}^{n \times d}$
- First Layer Weights $W_{xh}^{(1)} \in \mathbb{R}^{d \times h}$
- First Layer Hidden State Weights $W_{hh}^{(1)} \in \mathbb{R}^{h\times h}$
- First Layer Bias $b_h \in \mathbb{R}^{1 \times h}$
- First Layer Output, $H_t^{(1)} \in \mathbb{R}^{n \times h}$
- First Layer HIdden State, $H_{t-1}^{(1)} \in \mathbb{R}^{n \times h}$
- Second Layer Weights $W_{hh_2}^{(2)} \in \mathbb{R}^{h \times h_2}$ 
- Second Layer Hidden State Weights $W_{h_2h_2}^{(2)} \in \mathbb{R}^{h_2 \times h_2}$
- Second Layer Bias $b_{h_2} \in \mathbb{R}^{1 \times h_2}$
- Second layer Output, $H_t^{(2)} \in \mathbb{R}^{n \times h_2}$.
- Second Layer Hidden State, $H_{t-1}^{(2)} \in \mathbb{R}^{n \times h_2}$
- Third Layer Weights $W^{(3)}_{h_2a} \in \mathbb{R}^{h_2 \times a}$
- Third Layer Bias $b_a \in \mathbb{R}^{1 \times a}$
- Third layer output $A_t^{(3)} \in \mathbb{R}^{n \times a}$

> $h_i$ denotes the count of hidden units for the $ith$ hidden layer. $a$ is the count of output units. $n$ is batch size. $d$ is dimensionality of the input embeddings.

### Sec 1.

Assume a 2-layer Stacked RNN as:

```math

H^{(1)}_t = \phi(X^{(1)}_tW^{(1)}_{xh} + H^{(1)}_{t-1}W^{(1)}_{hh} + b^{(1)}_h)
\\[3mm]
H^{(2)}_t = \phi(H^{(1)}_tW^{(2)}_{hh_2} + H^{(2)}_{t-1}W^{(2)}_{h_2h_2} + b^{(2)}_{h_2})
\\[3mm]
A^{(3)}_t =  \text{softmax}(H^{(2)}_tW_{h_2a}^{(3)} + b_a^{(3)})

```

We can derive backpropagation for a single time step, $t = 2$, assuming $T = 2$ (total time steps / sequence length) (only computing $\partial W$, $\partial H$, and $\partial A$ omitting $\partial b$ for simplicity), where $H$ is initialized to a vector of $0$'s for the first time step:

```math

\frac{\partial L}{\partial Z_t^{(3)}} = A_t^{(3)} - \text{one-hot}(y) \hspace{1mm} \in \mathbb{R}^{n \times a}
\\[3mm]
\frac{\partial L}{\partial W^{(3)}_{h_2a}} = \left( \frac{\partial L}{\partial Z_t^{(3)}} \right) \left( \frac{\partial Z^{(3)}_t}{\partial W^{(3)}_{h_2a}}\right) = (H_t^{(2)})^T \cdot \frac{\partial L}{\partial Z_t^{(3)}}  \hspace{1mm} \in \mathbb{R}^{h_2 \times a}
\\[3mm]

\\[3mm]
\frac{\partial L}{\partial Z_{t}^{(2)}} =  \left( \frac{\partial L}{\partial Z_t^{(3)}} \right) \left( \frac{\partial Z^{(3)}_t}{\partial H^{(2)}_{t}}\right)\left(\frac{\partial H_t^{(2)}}{\partial Z_t^{(2)}} \right) =\left(\frac{\partial L}{\partial Z_t^{(3)}} \cdot (W_{h_2a}^{(3)})^T\right) \odot \phi'(Z_t^{(2)}) \hspace{1mm} \in \mathbb{R}^{n \times h_2}
\\[3mm]

\frac{\partial L}{\partial W_{hh_2}^{(2)}} =  \left( \frac{\partial L}{\partial Z_t^{(3)}} \right) \left( \frac{\partial Z^{(3)}_t}{\partial H^{(2)}_{t}}\right)\left(\frac{\partial H_t^{(2)}}{\partial Z_t^{(2)}} \right)\left(\frac{\partial Z_t^{(2)}}{\partial W_{hh_2}^{(2)}} \right)  + \left( \frac{\partial L}{\partial Z_t^{(3)}} \right) \left( \frac{\partial Z^{(3)}_t}{\partial H^{(2)}_{t}}\right)\left(\frac{\partial H_t^{(2)}}{\partial Z_t^{(2)}} \right)\left(\frac{\partial Z_t^{(2)}}{\partial H_{t-1}^{(2)}} \right)\left( \frac{\partial H_{t-1}^{(2)}}{\partial Z_{t-1}^{(2)}}\right) \left(\frac{\partial Z_{t-1}^{(2)}}{\partial W_{hh_2}^{(2)}}\right) = ((H_t^{(1)})^T \cdot \frac{\partial L}{\partial Z_t^{(2)}}) +  (H_{t-1}^{(1)})^T \cdot \left( \left( \frac{\partial L}{\partial Z_t^{(2)}} \cdot (W_{h_2h_2}^{(2)}) \right) \odot (\phi'(Z_{t-1}^{(2)})) \right) \hspace{1mm} \in \mathbb{R}^{h \times h_2}  

\\[3mm]
\frac{\partial L}{\partial W_{h_2h_2}^{(2)}} = (H_{t-1}^{(2)})^T \cdot  \left( \frac{\partial L}{\partial Z_t^{(3)}} \right) \left( \frac{\partial Z^{(3)}_t}{\partial H^{(2)}_{t}}\right)\left(\frac{\partial H_t^{(2)}}{\partial Z_t^{(2)}} \right)\left(\frac{\partial Z_t^{(2)}}{\partial W_{h_2h_2}^{(2)}} \right)  + (W_{h_2h_2}^{(2)})^T \cdot \left( \frac{\partial L}{\partial Z_t^{(3)}} \right) \left( \frac{\partial Z^{(3)}_t}{\partial H^{(2)}_{t}}\right)\left(\frac{\partial H_t^{(2)}}{\partial Z_t^{(2)}} \right)\left(\frac{\partial Z_t^{(2)}}{\partial H_{t-1}^{(2)}} \right) \left( \frac{\partial H_{t-1}^{(2)}}{\partial Z_{t-1}^{(2)}}\right) \left( \frac{\partial Z_{t-1}^{(2)}}{\partial W_{h_2h_2}^{(2)}}\right) = (H_{t-1}^{(2)})^T\left( (H_{t-1}^{(2)})^T \cdot  \frac{\partial L}{\partial Z_t^{(2)}} \right) + (W_{h_2h_2}^{(2)})^T \cdot \left( (H_{t-2}^{(2)})^T \cdot \left(\left( \frac{\partial L}{\partial Z_t^{(2)}} \cdot (W_{h_2h_2}^{(2)}) \right) \odot (\phi'(Z_{t-1}^{(2)})) \right)\right)  \hspace{1mm} \in \mathbb{R}^{h_2 \times h_2}  
\\[3mm]
\frac{\partial L}{\partial Z_t^{(1)}} = \left( \frac{\partial L}{\partial Z_t^{(3)}} \right) \left( \frac{\partial Z^{(3)}_t}{\partial H^{(2)}_{t}}\right)\left(\frac{\partial H_t^{(2)}}{\partial Z_t^{(2)}} \right)\left(\frac{\partial Z_t^{(2)}}{\partial H_{t}^{(1)}} \right)\left( \frac{\partial H_t^{(1)}}{\partial Z_t^{(1)}} \right) = \left( \frac{\partial L}{\partial Z_t^{(2)}} \cdot (W_{hh_2}^{(2)})^T\right) \odot \phi'(Z_t^{(1)}) \hspace{1mm} \in \mathbb{R}^{n \times h}
\\[3mm]
\frac{\partial L}{\partial W_{xh}^{(1)}} = \left( \frac{\partial L}{\partial Z_t^{(3)}} \right) \left( \frac{\partial Z^{(3)}_t}{\partial H^{(2)}_{t}}\right)\left(\frac{\partial H_t^{(2)}}{\partial Z_t^{(2)}} \right)\left(\frac{\partial Z_t^{(2)}}{\partial H_{t}^{(1)}} \right)\left( \frac{\partial H_t^{(1)}}{\partial Z_t^{(1)}} \right)\left( \frac{\partial Z_t^{(1)}}{\partial W_{xh}^{(1)}} \right) +\left( \frac{\partial L}{\partial Z_t^{(3)}} \right) \left( \frac{\partial Z^{(3)}_t}{\partial H^{(2)}_{t}}\right)\left(\frac{\partial H_t^{(2)}}{\partial Z_t^{(2)}} \right)\left(\frac{\partial Z_t^{(2)}}{\partial H_{t}^{(1)}} \right)\left( \frac{\partial H_t^{(1)}}{\partial Z_t^{(1)}} \right)\left( \frac{\partial Z_t^{(1)}}{\partial H_{t-1}^{(1)}}\right)\left(\frac{\partial H_{t-1}^{(1)}}{\partial Z_{t-1}^{(1)}}\right) \left( \frac{\partial Z_{t-1}^{(1)}}{\partial W_{xh}^{(1)}} \right) = (X_t^{(1)})^T \cdot \frac{\partial L}{\partial Z_t^{(1)}} + (X_{t-1}^{(1)})^T \cdot \left( \left( \frac{\partial L}{\partial Z_t^{(1)}} \cdot (W_{hh}^{(1)}) \right) \odot (\phi'(Z_{t-1}^{(1)})) \right) \in \mathbb{R}^{d \times h}
\\[3mm]
\frac{\partial L}{\partial W_{hh}^{(1)}} = (H_{t-1}^{(1)})^T \cdot \left( \frac{\partial L}{\partial Z_t^{(3)}} \right) \left( \frac{\partial Z^{(3)}_t}{\partial H^{(2)}_{t}}\right)\left(\frac{\partial H_t^{(2)}}{\partial Z_t^{(2)}} \right)\left(\frac{\partial Z_t^{(2)}}{\partial H_{t}^{(1)}} \right)\left( \frac{\partial H_t^{(1)}}{\partial Z_t^{(1)}} \right)\left( \frac{\partial Z_t^{(1)}}{\partial W_{hh}^{(1)}} \right) + (W_{hh}^{(1)})^T \cdot \left( \frac{\partial L}{\partial Z_t^{(3)}} \right) \left( \frac{\partial Z^{(3)}_t}{\partial H^{(2)}_{t}}\right)\left(\frac{\partial H_t^{(2)}}{\partial Z_t^{(2)}} \right)\left(\frac{\partial Z_t^{(2)}}{\partial H_{t}^{(1)}} \right)\left( \frac{\partial H_t^{(1)}}{\partial Z_t^{(1)}} \right)\left( \frac{\partial Z_t^{(1)}}{\partial H_{t-1}^{(1)}} \right) \left( \frac{\partial H_{t-1}^{(1)}}{\partial Z_{t-1}^{(1)}} \right) \left( \frac{\partial Z_{t-1}^{(1)}}{\partial W_{hh}^{(1)}}\right) = (H_{t-1}^{(1)})^T \cdot \left((H_{t-1}^{(1)})^T \cdot \frac{\partial L}{\partial Z_t^{(1)}} \right ) + (W_{hh}^{(1)})^T \cdot \left( (H_{t-2}^{(1)})^T \left( \left( \frac{\partial L}{\partial Z_t^{(1)}} \cdot  (W_{hh}^{(1)})\right) \odot  (\phi'(Z_{t-1}^{(1)})) \right) \right)  \in \mathbb{R}^{h \times h}
```

### Sec 2.

Now for multiple timesteps, $t \in [1, T = 2]$, where $T$ is total time steps, equivalent to input sequence length, we can compute the total gradients as a summation of the gradients at each $t$:

```math

\frac{\partial L}{\partial Z^{(3)}} = \frac{1}{T}\sum_{t=1}^{T} \frac{\partial L}{\partial Z_t^{3}} 
\\[3mm]
\frac{\partial L}{\partial W_{h_2a}^{(3)}} = \frac{1}{T} \sum_{t=1}^{T} \left(\frac{\partial L}{\partial W_{h_2a}^{(3)}} \right)_{(t)}
\\[3mm]
\frac{\partial L}{\partial Z^{(2)}} = \frac{1}{T} \sum_{t=1}^{T} \frac{\partial L}{\partial Z_t^{2}} 
\\[3mm]
\frac{\partial L}{\partial W_{hh_2}^{(2)}} = \frac{1}{T} \sum_{t=1}^{T} \left(\frac{\partial L}{\partial W_{hh_2}^{(2)}} \right)_{(t)}
\\[3mm]
\frac{\partial L}{\partial W_{h_2h_2}^{(2)}} = \frac{1}{T} \sum_{t=1}^{T} \left(\frac{\partial L}{\partial W_{h_2h_2}^{(2)}} \right)_{(t)}
\\[3mm]
\frac{\partial L}{\partial Z^{(1)}} = \frac{1}{T} \sum_{t=1}^{T} \frac{\partial L}{\partial Z_t^{1}} 
\\[3mm]
\frac{\partial L}{\partial W_{xh}^{(1)}} = \frac{1}{T} \sum_{t=1}^{T} \left(\frac{\partial L}{\partial W_{xh}^{(1)}} \right)_{(t)}
\\[3mm]
\frac{\partial L}{\partial W_{hh}^{(1)}} = \frac{1}{T} \sum_{t=1}^{T} \left(\frac{\partial L}{\partial W_{hh}^{(1)}} \right)_{(t)}
```

This is all generalizable to any $T \in [1, \infty]$ but the original forward pass (see Sec 1.) will begin to rely on a larger set of $t$ such that there will be more gradient factors in the chain rule during backpropagation, purely for a single time step $t$, as we'll have to backpropate through a larger set of hidden states $H_t$ to get the gradients for a single $Z$ or $W$.

In practice, this leads a greater risk of vanishing or exploding gradients, which can be dependent on the magnitude of $W$ or $Z$.

```math
\frac{\partial L}{\partial W_{hh}^{(1)}} = \sum_{t=1}^{T} \left( H_{t-1}^{(1)} \right)^T \cdot \left( \prod_{k=t}^{T} \frac{\partial Z_k^{(3)}}{\partial H_k^{(2)}} \cdot \frac{\partial H_k^{(2)}}{\partial Z_k^{(2)}} \cdot \frac{\partial Z_k^{(2)}}{\partial H_k^{(1)}} \cdot \frac{\partial H_k^{(1)}}{\partial Z_k^{(1)}} \right) \cdot \frac{\partial Z_t^{(1)}}{\partial W_{hh}^{(1)}}

```

This can clearly be seen here, as for multiple $t$, we rely on a deeper product of gradients via the chain rule.

A solution to this is truncated gradients to be computed through $T - \tau$ time steps, where $\tau < T$ rather than $T$. This leads to an RNN becoming less complex, less prone to overfitting, acting as a form of regularization, as the model is only learning based on gradients up to time $T - \tau$ .

### more on vanishing / exploding gradients

A common issue presented are vanishing / exploding gradients.

A forward pass through an RNN goes through $T$ layers for a single token in the sequence. For the backward pass, we backpropagate through $T$ layers for all input sequences, such that we ultimately have a chain of matrix-products (chain rule) of length $\mathcal{O}(T)$, dependent on the sequence length of the input.

> *Depth of backpropagation through time (or the amount of factors in the chain of matrix-products) can be defined as $sT$ where $s$ is the sequence length and $T$ is the count of layers.*
>
> *Not simply $T$ as would be in a feed-forward neural network*.


If the value of the backpropagated gradients, $\partial Z$, are too small or too large, we can easily be led into encountering exploding or vanishing gradients, which can worsen as the model gets deeper.

A solution to  ( not regarding LSTMs or GRUs ) is to gradient clip, as:


```math

\min(1, \frac{\theta}{||g||})g

```

where $g = g$ if $\theta > g$ or $g = \frac{\theta}{||g||}g$ if $g < \theta$.

This effectively constrained $g$ to be within the $\text{L}2$ ball defined by the constant $\theta$.

<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>

---

# whiteboard for building a 2-layer RNN in numpy

## two layer RNN

### first layer

$h$ = hidden size for 1st layer.

- Input $X_t \in \mathbb{R}^{n \times d}$
- First Layer Weights $W^{(1)} \in \mathbb{R}^{d \times h}$ 
- First Layer Hidden State Weights $W_{h}^{(1)} \in \mathbb{R}^{h \times h}$
- First Layer Bias $b \in \mathbb{R}^{1 \times h}$
- First layer Output, $H_t^{(1)} \in \mathbb{R}^{n \times h}$.
- First Layer Hidden State, $H_{t-1}^{(1)} \in \mathbb{R}^{n \times h}$

```math

H_t = \phi(X_tW^{(1)} + H_{t-1}^{(1)}W_h^{(1)} + b)

```

### second layer

$h_2$ = hidden size for 2nd layer.

- Input $H_t^{(1)} \in \mathbb{R}^{n \times h}$
- Second Layer Weights $W^{(2)} \in \mathbb{R}^{h \times h_2}$ 
- Second Layer Hidden State Weights $W_{h}^{(2)} \in \mathbb{R}^{h_2 \times h_2}$
- Second Layer Bias $b \in \mathbb{R}^{1 \times h_2}$
- Second layer Output, $H_t^{(2)} \in \mathbb{R}^{n \times h_2}$.
- Second Layer Hidden State, $H_{t-1}^{(1)} \in \mathbb{R}^{n \times h_2}$

```math
H_t^{(2)} = \phi(H_t^{(1)}W^{(2)} + H_{t-1}^{(2)} W_h^{(2)} + b)
```

### now for multiple time steps,

```math
\mathcal{X} = \begin{bmatrix} X_1 & X_2 & \dots & S \end{bmatrix}

\\[3mm]

\mathcal{X} = \begin{bmatrix} 
\begin{bmatrix} .2 & .3 & \dots & d \end{bmatrix} & 
\begin{bmatrix} .5 & .2 & \dots & d \end{bmatrix} & 
\cdots & 
\begin{bmatrix} .9 & .1 & \dots & d \end{bmatrix}&
S
\end{bmatrix} \in \mathbb{R}^{1 \times d \times S}

```

where $d$ is the dimensionality of the embedding space and $S$ is the fixed size sequence length.

### for multiple time steps and larger batch size,
<br/>

```math
\mathcal{X} = \begin{bmatrix} X^{(1)}_1 &  X^{(1)}_2 & \dots & X_S^{(1)} \\ \vdots & \vdots & \ddots & \vdots \\ X^{(n)}_1 & X^{(n)}_2 & \dots & X_S^{(n)}  \end{bmatrix}
```
<br/>

```math

\mathcal{X} = \begin{bmatrix} 
\begin{bmatrix} .2 & .3 & \dots & d \end{bmatrix}^{(1)}_1 & 
\begin{bmatrix} .5 & .2 & \dots & d \end{bmatrix}^{(1)}_2 & 
\cdots & 
\begin{bmatrix} .9 & .1 & \dots & d \end{bmatrix}^{(1)}_S \\
\vdots & \vdots & \ddots & \vdots \\
\begin{bmatrix} .4 & .3 & \dots & d \end{bmatrix}^{(n)}_1 & 
\begin{bmatrix} .5 & .4 & \dots & d \end{bmatrix}^{(n)}_2 & 
\cdots & 
\begin{bmatrix} .9 & .7 & \dots & d \end{bmatrix}^{(n)}_S
\end{bmatrix} \in \mathbb{R}^{n \times d \times S}

```

where $n$ is the batch size, $d$ is the dimension of the embedding space, and $S$ is the sequence length.


```python
n = batch_size
d = embedding_space_dims
s = sequence_length

X.shape = (n, d, s)
```

`rnn.a` will be constructed as as a list of lists, wheres `rnn.a` is a list and each $ith$ list within `rnn.a` will be correspondent to the $ith$ layer. each index within the lists within `rnn.a` will be correspondnet to the time step. 

for the $ith$ list within `rnn.a`, the $jth$ index within the $ith$ list corresponds to time step $t$. 

denoting the $ith$ list in `rnn.a` as `a_i`, to index the $jth$ time step, we'll have `a_i[j]`

`i = layer` <br/>
`j = time_step`

or more generally, `rnn.a[i][j]`

each index of `rnn.a` is for a given layer and within each indexed list lies activations at multiple time steps.

```python
rnn.a = [
    [[0.1, 0.2], [0.3, 0.4]],  # Layer 1 (2 time steps)
    [[0.5, 0.6], [0.7, 0.8]],  # Layer 2 (2 time steps)
]
```

i finished the correct initialization for the `rnn.a` and `rnn.z`

- [ ] Fix Forward pass given new shape of `rnn.a` and `rnn.z`
