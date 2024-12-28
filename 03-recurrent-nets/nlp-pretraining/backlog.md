### Word Vectors

Individual words or subsets of them can be represented as word vectors $\in \mathbb{R}^n$, where $n$ denotes the number of individual features that make up a word vector.

Say we denote a word-vector as $w_i$, where $i$ is the $ith$ vector in the entire vocabulary set, meaning $w_i \in \mathcal{V}$.

The larger $n$ is, the more representative the embedding space, $\mathbb{R}^n$ is, and therefore it becomes easier to distinguish a larger subset of word vector $\in \mathcal{V}$. This becomes particularly important for large sets of $\mathcal{V}$.

> *Bias-Variance trade-off plays a huge role here, the higher $\mathbb{R}^n$ is, the higher likelihood the embedding model will overfit, unable to properly place unseen words near similar positions in the $\mathbb{R}^n$ embedding space to similar words.*
>
> *Same with the curse of dimensionality, we'll need mroe data the larger $n$ becomes.*

Each dimension can account for an arbitrary feature of a given word, such as tense, singular v plural, gender, connotation, etc.

### One-hot word-vectors suck

A one-hot embedding vector is represented as:

```math

\vec{w_1} = \begin{bmatrix}1 \\0\\ 0 \\\vdots \\0\end{bmatrix}
\\[3mm]
\vec{w_2} = \begin{bmatrix}0 \\1\\ 0 \\\vdots \\0\end{bmatrix}

```

If the dot product of two vectors represents their similarity, the dot product of one-hot embeddings will always be $0$, as each $w_i \in \mathcal{V}_{\text{onehot}}$ is orthogonal to each other.

In this case, we have no means to encode the similarity of different word-vectors in an embedding space.

Similarity metrics, such as the aforementioned dot product or Cosine Similarity won't work.

### SVD

We can use SVD to compose word embeddings by accumulating word co-occurence counts (co-variate matrix?) and then performing SVD to get $X = U\Sigma V^T$, where the rows of $U$ denotes our word embeddings.

$X$ can be constructed as a co-occurrence or affinity matrix.

Then you can simply solve for the embeddings by computing $\text{SVD} = U\Sigma V^T$, where $U$ is the embedding matrix.

Reducing the singular values, $\sigma \in \Sigma$, alongside $u_i$, reduces dimensionality of the embedding space.

### Word2Vec

> Note that word2vec models take the form of a neural network.

Instead of computing SVD to generate word-embeddings, we can train a model to encode the word vectors as it's parameters, given an objective function.

Contains two, **continuous bag of words (CBOW)** and **skip-gram**.

The former predicts the probability of a center word from surrounding context while skipgram predicts the probability of context words given a center word.

And there are two training methods, **negative-sampling** and **hierarchical softmax**.

quick detour --

### unigram, bigram, n-gram

unigram models assume that the probability of a given word is independent of previous words such that $P(w_1, w_2, \dots, w_n) = P(w_1) \cdot P(w_2) \cdot \dots \cdot P(w_n)$.

bigram models assume that the probability of a given word is dependent on the current - 1 word such that $P(w_1, w_2, \dots, w_n) = P(w_1) \cdot P(w_2 | w_1) \cdot \dots \cdot P( w_{n} | w_{n-1})$ 

```math

P(w_1, w_2, \dots, w_3) = \prod_{i = 2}^n P(w_i | w_{i - 1})

```

> (word2word aka co-occurrence matrices seem to fit best for bigram language modelling, as the capture the relationship between a pair of words, which is what bigram modelling is best for.)

n-gram models assume that the probability of a given wrod is dependent on the $n$ previous words to the current word such that, $P(w_1, w_2, \dots, w_n) = P(w_1) \cdot P(w_2 | w_1) \cdot P(w_3 | w_2 , w_1) \dots P(w_n | \dots , w_3 , w_2 , w_1)$

> key takeaway, is that n-gram modelling denotes the concept of how many words prior to the current word a fixed window size language model considers to model $P(w_1, \dots, w_n)$

### CBOW

One approach is to take a set of context words and predict the center word.

Given

```math

{\text{\{"My, first, and, name, is, 'John'}"\}}

```

we try to predict the word "$\text{last}$", in between $\text{"and"}$ + $\text{"name"}$.

We train the model on one-hot word vectors, say $x^{(c)}$, which denotes the context or input sequence to predict $y$, the target word represented as a one-hot vector.

We create two matrices, $V$ and $U$, from the original one-hot encodings.

```math
V \in \mathbb{R}^{n \times |V|} \\[3mm] U \in \mathbb{R}^{|V| \times n}
```

$V$ is the input matrix. $n$ is the dimensionality of the word embeddings and $|V|$ is the size of the vocabulary.

The $ith$ column of $V$ denotes the embedding of a singular word, as $\vec{v}_i$ for word $w_i$, when it's used as a context word (input).

$U$ is the output matrix, where each row of $U$ is the word's outpu embedding when it's used as a target word (output).

1. Generate one hot word vectors, for input context size of $m$ as:

    ```math
    x^{(c - m)}, \dots, x^{(c - 1)}, x^{(c + 1)}\dots, x^{(c + m)} \in \mathbb{R}^{|V|} 
    ```

2. Generate $V$

    - $v_{c-m}, v_{c - m + 1}, \dots , v_{c + m} \in V$ are the context word vectors. Index $c$ is the center word and is not considered $\in V$ when attempting to predict the center word.

    - $x_{c-m}, x_{c - m + 1}, \dots , x_{c + m}$ are the one hot vectors for each of the context words (embeddings prior to generating $V$).

    - $V$ is the final matrix of input embeddings.

    - We randomly initialize $V$.

    - Given a vocabulary of say:
  
        ```
        {'table': 0, 'mouse': 1, 'The': 2, 'barked': 3, 'loudly': 4, 'mat': 5, 
        'on': 6, 'under': 7, 'dog': 8, 'chased': 9, 'sat': 10, 'ran': 11, 
        'cat': 12, 'the': 13}
        ```

        we can generate $V$ (randomly) as (just an example of what each row indicates):

        ```
        V = [
        [0.1, 0.2, 0.3],  # 'table'
        [0.4, 0.5, 0.6],  # 'mouse'
        [0.7, 0.8, 0.9],  # 'The'
        ]
        ...
        ```
  
3. Average the vectors to get $\hat{v}$, as:

    ```math

    \hat{v} =  \frac{v_{c-m}, v_{c - m + 1}, \dots , v_{c + m}}{2m} \in \mathbb{R}^n

    ```

    note that we normalize by $2m$ as we're considering the context in front and behind the center word.

    We choose a given $v_i$ by the input / output data. Say given an input context of 'table' and 'the', with center word as 'brown' we find $\hat{v}$ as the average of the vectors $\in V$ that match the index of 'table' and 'the' relevant to the aforementioend vocabulary set (see #2)

4. Generate a score vector as

    ```math

    z = U\hat{v} \in \mathbb{R}^{|V|}

    ```

    $\in \mathbb{R}^{|V|}$ as $U$ is $\in \mathbb{R}^{|V| \times n}$ and $\hat{v}$ is $\in \mathbb{R}^n$. Then it's easy to see that the matrix multiplication of $U\hat{v}$ generates a vector which is $\in \mathbb{R}^{|V|}$

    $U$ are the parameters of the neural network.

5. Compute final probabilites as $\hat{y} =  \text{softmax(z)}$

    The final probabilities $\hat{y}$ should ideally match the true probabilties, $y$.

So the final true probabilities $y$ are the one-hot encodings for the ground truth "probabilities". 

Then $V$ are the input embeddings for the context words and $U$ are the learned parameters, denoting the embeddings for the center words. Both $V$ and $U$ are $\in \mathbb{R}^{n \times |V|}$ (though one has a transposed shape).

We're essentially aiming to have $\hat{y} = y$ (though will never be truly the case) such that $U$ gets trained to represent the center word embeddings.

To train the model, the cross entropy loss is used as the objective:

```math

\mathcal{L} = \sum_{i = 1}^{|V|}- y_i \cdot \log(\hat{y}_i)
\\[3mm]
\min \mathcal{L}
```

trained via stochastic gradient descent, using the general update rule.

> Note that we're also actively training the inputs, $V$, in addition to $U$. This is as we want to generate embeddings for context words, in addition to the embeddings for the center words. You simply compute $\frac{∂\mathcal{L}}{∂V}$ and use the general update rule.
>
> I originally thought this would result in $∂L = 0$, but in reality it's as: $\frac{∂L}{∂V} = \frac{∂L}{∂Y}\frac{∂Y}{∂\hat{v}} = \frac{∂L}{∂Y}{x}$ if we assume a simple function: $Vx = Y$, where $V$ is the input embedding.
>
> Also note that input embeddings are actually $\vec{\hat{v}}$ or ${\hat{V}}$.