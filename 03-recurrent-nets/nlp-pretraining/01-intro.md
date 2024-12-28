# Intro to NLP

- [ ] will be poast???¿ i think so.

## Word Embeddings

- One-hot vectors suck. There's no means to compute the cosine similarity or other similarity metrics.

    I want a glass of orange "**juice**"\
    I want a glass of apple _____.

    One-hot vectors of "orange" and "apple" have no notion of similarity to each other, given that they are orthogonal -- you can't reliably compute the dot product or cosine similarity to gauge the magnitude of similarity.

- Instead, you can learn a set of Word Embeddings (non-one hot manner, but instead featurized numerical representations).

    ```math

    \text{man} = \begin{bmatrix}-1 \\ .01 \\ .03\end{bmatrix}
    \\[3mm]
    \text{woman} = \begin{bmatrix} 1 \\ .02 \\ .02 \end{bmatrix}

    ```

    In the above example, the numerical values represent the learnt features for the given sample "man" or "woman". 

    $\in \mathbb{R}^3$

    The learned features, from an outside perspective, aren't interpretable (lack of feature labelling), but to the model that has learnt them, they are very interpretable. 

    Though you can have emergent properties.
    Say we have $\vec{\text{king}}$ and we compute:

    ```math

    \vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} = \vec{\text{queen}}

    ```

    we get queen, meaning we may be able to identity which feature belongs to 'gender' by taking a look at the major difference in the $\vec{\text{man}}$ and $\vec{\text{woman}}$ vectors. Though this isn't guaranteed.

    Now that we have non-one hot embeddings, we can compute similarity metrics, such as the cosine simliarity.

    While models don't explicitly compute similarity metrics when leveraging embeddings, the geometry of the embedding space still matters. To predict the next word, similar embeddings compute similar output logits, $z$, such that the value of the output probability (softmax) or the loss will be similar given $\vec{a}$ and $\vec{b}$ if they have a high $\cos$ similarity.

## making use of word embeddings

We learn an **embedding matrix**, $E$, of which the column vectors, $e_i$, are the individual word embeddings in $\mathbb{R}^n$.

Then the matrix becomes an $n \times |V|$ matrix, where $|V|$ is the size of the vocabulary.

### training a neural language model / learning word embeddings

Assume we have a set of one hot vectors representing a set of words $\in |V|$, which are $\in \mathbb{R}^{|V| \times 1}$

```math

\vec{o_1}\\[3mmm]\
\vec{o_2} \\[3mm]
\vec{o_3}

```

We can multiply this by an embedding matrix, $E \in \mathbb{R}^{|V| \times n}$ (initially random prior to training), 

```math

E^T\vec{o_i}

```

to extract the embedding vectors:


```math

\vec{e_1} \\[3mmm]
\vec{e_2} \\[3mm]
\vec{e_3}

```

which are $\in \mathbb{R}^{n \times 1}$

These embedding vectors are then fed into a language model for training. The final output is decided via a Softmax activation.

Note that in language models, we typically automate the above steps by including an embedding layer with weights $E$, as we need to train the emebddings via gradient descent.

The architecture looks as: 

```math

\text{Input | E | } \hspace{1mm} E \hspace{1mm} \in \mathbb{R}^{|V| \times n} \rightarrow \text{Hidden Layer | } W_1, b_1 \hspace{1mm} | \hspace{1mm} W_1 \in \mathbb{R}^{\text{n}\times m}  \rightarrow \text{Softmax Layer | } W_2, b_2 \hspace{1mm} | \hspace{1mm} W_2 \in \mathbb{R}^{m \times |V| } \rightarrow \hat{y} \in \mathbb{R}^{|V| \times \text{batch}_{\text{size}}}


```

I/O shape flow for each layer is as:


```math

E^To \in \mathbb{R}^{n \times \text{batch}_{\text{size}}} \rightarrow W_1^To^{(1)} \in \mathbb{R}^{m \times \text{batch}_{\text{size}}} \rightarrow W_2^To^{(2)} \in \mathbb{R}^{|V| \times \text{batch}_{\text{size}}}

```

ultimately, this model learns word-embeddings pretty well!

We're able to extract the parameters ($E$) as the embedding matrix, as the model ultimately aims to learn the features of the inputs $\vec{o}_i$ or $O$ within it's weights.

The embedding matrix then slowly becomes representative of the true embeddings of the vocabulary, given that the model is sufficiently trained to predict the next word.

This works if we're explicitly trying to train a language model, to predict the next-word given a context size of $n$, while generating embeddings simultaeneously.

But, if we aren't attempting to train a next-word predictor, we can simply train a model with input context to be **$n$ surrounding words**, **last-1 word**, or **a nearby context word**.

## ¡word2vec!

### **Skipgrams** 

Say we have the vocabulary 

```math

V = \{\text{I'm Geoffrey Hinton, the godfather of deep learning.}\}
```

with the context word being $\text{godfather}$

A skipgram model aims to output the probability for all target words in a vocabulary, $V$, in context of $\text{godfather}$. It's called skipgram as we typically skip a couple of words from the context word to yield a target word.

```math

\text{input | godfather } \rightarrow \text{skipgram} \rightarrow \hat{y_i}

```

where $y_i$ is the probability for $ith$ target word, $t_i$

More specifically:

```math

E^To_{\text{godfather}} = e_{\text{godfather}} \rightarrow W^Te_{\text{godfather}} = z \rightarrow \text{softmax(z)} = \hat{y}

```

where 

- $E \in \mathbb{R}^{|V| \times n}$ as the embedding matrix ($|V|$ is size of vocab and $n$ is the size of the hidden layer or embedding space.)
- $W \in \mathbb{R}^{n \times |V|}$
- $\hat{Y} \in \mathbb{R}^{|V| \times \text{batch}_{\text{size}}}$, the output probability matrix for all words $\in V$

### Hierarchical Softmax

In a very large $V$, the softmax classifier becomes very computationally expensive, especially when computing gradients.
Given a very large output vector, $\hat{y} \in \mathbb{R}^{|V| \times 1}$, the computation of $\frac{\mathcal{∂L}}{{∂\hat{y}}}$ and some subsequent gradients at earlier layers can become computationally expensive.

Instead we can use softmax as a binary tree -- hierarchical softmax -- where each node represents a subset of the vocabulary, $v \in V$ and the root node is a given word.

Say the left node is $v_l$ and the right node is $v_r$.

At each node of the tree, we compute $\sigma$ or sigmoid activation. If $\sigma(z) < 1$, we go left -- otherwise we go to the right subset.

This is done until we reach a root node, where lies the predicted word.

### Negative Sampling

Similar to Hierarchical Softmax, this aims to minimize the computational cost of training models on a large $V$.

Given a target word, $w_t$ and (set of) context word $w_c \in W_c$, we compute a binary classification for each possible output. 

Then, we choose a subset of words, $k+1$ where $k$ is the number of negative samples (hyperparameter) and $1$ comes from the target word. 

In essence the subset of words includes the target word + any random $k$ set of words. The larger $k$ is the more computationally expensive the model will be. The inverse is true for smaller $k$.

Then, we compute backprop (using binary cross entropy loss) only with respect to the $k+1$ subset of words.

> Note that the gradients for each individual k+1 are summed into a singular gradient vector.

---

<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>
<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>
