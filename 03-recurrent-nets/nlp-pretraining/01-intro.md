# Intro to NLP


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

Note that in language models, we typically automate the above steps by including an embedding layer with weights $E$, as we need to train the embeddings via gradient descent.

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

> Note that this language model can take in input sequences of words larger than $1$ as well, as $n-1$ where $n$ is your context window.
> 
> Assume we have two words, represented by the one hot embeddings $\vec{o_1}$ and $\vec{o_2}$. They're separately multiplied with $E$, to extract $\vec{e_1}$ and $\vec{e_2}$. Afterwards, $\vec{e_0}$ and $\vec{e_1}$ are concatenated as:
>
> ```math
> \vec{x} = \begin{bmatrix} \vec{e_1} \\ \vec{e_2} \end{bmatrix} = \begin{bmatrix} e_{11} \\ e_{12} \\ e_{13} \\ e_{21} \\ e_{22} \\ e_{23} \end{bmatrix}
> ```
> into a single vector. 

This works if we're explicitly trying to train a language model, to predict the next-word given a context size of $n$, while generating embeddings simultaeneously.

> Note that the neural probabilistic language model can only be trained to a fixed-window size, meaning a specific $n$-gram task at once.

But, if we aren't attempting to train a next-word predictor, we can simply train a model with input context to be **$n$ surrounding words**, **last-1 word**, or **a nearby context word**.

## ¡word2vec!

### **Skipgrams** 

Say we have the sequence:

```math

S = \{\text{I'm Geoffrey Hinton, the godfather of deep learning.}\}
```

with the center word being $\text{godfather}$, serving as the input to the model.

> Skipgram takes in a single word as input.

Assume window size (for prediction) is 2 words, $k = 2$.

A skipgram model aims to output the probability for all target words in a vocabulary, $V$ (where $||V|| >> ||S||$), with the goal of correctly predicting the nearest words, in this case with a window size of $2$ -- $\text{the}$ and $\text{of}$. 

```math

\text{input | godfather } \rightarrow \text{skipgram} \rightarrow \hat{y} \in \mathbb{R}^{|V|} \rightarrow \argmax(\hat{y}) \in \mathbb{R}^{k}

```

where $\hat{y}$ is the vector modelling the softmax probabilities for the vocabulary $V$, given the center word, "godfather".

More specifically:

```math

E^To_{\text{godfather}} = e_{\text{godfather}} \rightarrow W^Te_{\text{godfather}} = z \rightarrow \text{softargmax}_{window_{size}}(z) = \hat{y}_{top-k} \in \mathbb{R}^{k}

```

where 

- $E \in \mathbb{R}^{|V| \times n}$ as the embedding matrix ($|V|$ is size of vocab and $n$ is the size of the hidden layer or embedding space.)
- $W \in \mathbb{R}^{n \times |V|}$

$W$, during training, gets tuned to represent the context word embeddings, such that the parameters, $E$ and $W$, represent two sets of embedding vectors.

For a more specific explanation onto why,

Say we're inputting a single embedding vector $\vec{e}$, extracted from $E$, to feed into the matmul with $W^T$.

Relying on the magnitude of the dot product as an interpretation for similarity (large positive, more similarity between two vectors), the larger the magnitude a given element of the output of the matmul is, the greater the similarity between $\vec{e}$ and a row of the embeddings $W^T$ are (denoted by a larger numerical value relative to other given elements in the output logits). 

From here, it's easy to see how the softmax is able to produce higher $P$ for similar words, as higher $P$ corresponds to a greater logit element in the logit vector, relative to its other elements.

Given that we're inputting a **center** word for the matmul with $W^T$, during backpropagation, $W^T$ gets trained to recognize the proper weights -- or equivalently embeddings -- for the context words. If dot product with a row of $W^T$ and $\vec{e}$ resulted in a higher logit, leading to an **incorrect** prediction, it's easy to see that the $∂\mathcal{L}$ and correspondingly, the update rule, will change the corresponding row embedding over multiple iterations to a more suitable value.

Then, intuitively, as we're backpropagating the gradient from the hidden layer to the embedding layer, $E$ gets trained to match the context word embeddings, $W$, to get the proper output predictions for the context word, over multiple iterations.

And they must, otherwise the model wouldn't mechanically function well enough to predict the proper context words. This naturally results in $E$ being center word emebddings (as that's our input) and $W$ being context words embeddings.

> Neural networks, are just math.

### CBOW

Unlike Skipgram, CBOW aims to predict the center word give context words.

The input to CBOW models is constructed as:

```math

E^T{O} = \hat{E} \\[3mm] \vec{x} = \frac{1}{k}\sum_{i = 1}^k {\hat{e}}_{i}

```

where $\vec{e_i}$ is the $ith$ vector $\in \hat{E}$, $k$ is the total count of context words, $O \in \mathbb{R}^{|V| \times n}$ is the matrix of one hot embeddings for the input context words ($n$ is the sequence length / context word count), and $E$ is the context word embeddings.

We feed $\vec{x}$ into the hidden layer with weights $W^T$, as:

```math

W^T\vec{x} = Z

```

to then get output probabilities for the center word as $\text{softmax}(Z)$.

Given that we're only predicting a singular center word, we simply extract the predicted word as an $\argmax()$ with $k = 1$.

Just as prior, we can derive the intuitive informal "proof" (because how formal / rigorous can you get with neural nets without insane complexity...) for the construction / training of the embeddings. It works the same way.

### Differences

- CBOW has faster training time (more ∂'s and weight updates)
- Skipgram has slower training time but can capture better relationships between rare words as the input is the average of relevant $\vec{e_i}$, such that gradients get distributed to correpsonding averaged $\vec{e_i}$'s.
  
    > feel free to derive it. great exercise.

### Negative Sampling

Similar to Hierarchical Softmax, this aims to minimize the computational cost of training models on a large $V$.

Given a target word, $w_t$ and (set of) context word $w_c \in W_c$, we compute a binary classification for each possible output. 

Then, we choose a subset of words, $k+1$ where $k$ is the number of negative samples (hyperparameter) and $1$ comes from the target word. 

In essence the subset of words includes the target word + any random $k$ set of words. The larger $k$ is the more computationally expensive the model will be. The inverse is true for smaller $k$.

Then, we compute backprop (using binary cross entropy loss) only with respect to the $k+1$ subset of words.

> Note that the gradients for each individual k+1 are summed into a singular gradient vector.

## GloVe

### co-occurence matrices.

Given

- "I like deep learning."
- "I like NLP."
- "Deep learning is fun."

The vocabulary $V = \{ \text{I}, \text{like}, \text{deep}, \text{learning}, \text{NLP}, \text{is}, \text{fun} \}$, which has a size of 7.

The context window size is 1, meaning each word in the corpus has its immediate neighbors as context words.

The global co-occurrence matrix $X$ is defined as:

$$
X_{i,j} = \text{count of how many times word } j \text{ appears in the context of word } i.
$$

The co-occurrence matrix $M$ is:

```math
X =
\begin{pmatrix}
0 & 2 & 0 & 0 & 0 & 0 & 0 \\
2 & 0 & 1 & 0 & 1 & 0 & 0 \\
0 & 1 & 0 & 2 & 0 & 0 & 0 \\
0 & 0 & 2 & 0 & 0 & 1 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 & 0 & 1 & 0
\end{pmatrix}

```

Where the rows and columns represent the words in the vocabulary in the following order:


```math

\text{\{{I,like,deep,learning,NLP,is,fun}\}}
```

We're essentially mapping how many times word $i$ appears in context of $j$, given the prior sequences defined above.

You can see this in terms of context and target words, where the rows or an $ith$ word is a given target word and columns or the $jth$ word is the context word.

### alg

GloVe makes use of $X$ to optimize the following function:

```math

\min \sum_{i = 1}^{|V|}\sum_{j = 1}^{|V|} f(X_{ij}) (u_j^Tv_i - \log(X_{ij}))^2

```

where

- $|V|$ is the size of the vocabulary
- $u_j$ is the $j$th column vector $\in U$, the context word embedding matrix
- $v_i$ is the $i$th row vector $\in V$, the target word embedding matrix.
- $u_j^Tv_i$ is the predicted co-occurence value for the word pair, $i, j$.
- $f(X_{ij})$ is a weighted function that equals $0$ if $X_{ij} = 0$ to avoid computing an undefined value when $X_{ij} = 0$ in the $\log$.
  - $f(X_{ij}) = \left( \frac{X_{ij}}{X_{\text{max}}} \right)^\alpha \quad \text{for} \quad X_{ij} < X_{\text{max}}$
    - $\alpha$ is a hyperparameter, which yields higher $f$ for higher $\alpha$, alongisade a higher loss value, and inversely for lower $\alpha$.
  - The function has higher values for more common $X_{ij}$, as it's closer to $X_{max}$ and therefore $f \rightarrow 1$. Inversely the function has lower values for less common $X_{ij}$.
  - Therefore when we compute the loss for a more common $X_{ij}$ or a more common word such as "the", we'll get a higher value when compared for a more rare $X_{ij} \in V$, such as "Accismus"

This is simply trained via gradient descent and the update rule.


## subword embeddings

- 'helps', 'helped', and 'helping' are inflected forms of the same word, 'help'. they are the more complex form of the *lemma*, 'help'.
- 'dog' and 'dogs' has the same relationship as 'cat' to 'cats'
- 'girl' and 'girlfriend' has the same relationship as 'boy' and 'boyfriend'

### fasttext

In word2vec, different inflected forms of a given lemma are represented directly by different vectors, without similar parameters. 

Therefore, the relationship between both the lemma and it's various inflected forms (which can span many different words, depennding on the language. for some words, spanish has 40+), which are mapped by the embeddings $\in \mathbb{R}^n$ don't have similarly encoded relationships.

> The same goes for GloVe

There are a few issues with this:

- Treating inflected forms as different vectors can lead to a non-trivial probability $P$ that the inflected forms don't have a high $\frac{\vec{x_1}\vec{x_2}}{||x_1||\cdot||x_2||}$, where $P$ is higher when a given inflected form is less common.
  - This typically only occurs when the model is trained on sequences where the context doesn't make use of rare inflected forms, such that some $x_i$ get trained but other inflected forms related to $x_i$ don't get trained and end up with a $\cos \theta << 1$
- Some languages have hundreds or thousands of inflected forms which increases computation / memory requirements for training and storing embeddings, leading to unneeded compelxity.

To use morphological information, meaning the information coming from the form / underlying composition of a given thing, to train embeddings, fastText proposes subword embeddings, where each subword embedding is a sequence of $n$ consecutive characters, strictly a portion of a given word.

To obtain a subword:

- Add '< >' to the end / beginning of the word (surround it).
- Then extract, character level $n$-grams from the word
  - if $n = 3$, for the word "embedding" we have, ['<em', 'emb', 'mbe', 'bed', 'edd', 'ddi', 'din', 'ing', 'ng>'] and the special subword "$\text{<}$where$>$" (to be treated as another vector to be trained as an embedding)
  - Note that $n$ can be any integer. The [paper](https://arxiv.org/pdf/1607.04606) recommends $n \in [3, 6]$.
  - Say we do have $n \in [3, 6]$ -- fastText will generate all possible subwords for a given word using all $n \in [3,6]$. For example if we have the word "embedding", the following are generated:

    ```math

    \text{Length 3 subwords: }  \{ \langle \text{em}, \text{emb}, \text{mbd}, \text{bdi}, \text{din}, \text{ing}, \text{ng} \rangle \} 
    \\[3mm]
    \text{Length 4 subwords: }  \{ \langle \text{emb}, \text{embe}, \text{mbed}, \text{bedd}, \text{eddi}, \text{ddin}, \text{ding}, \text{ing} \rangle \} 
    \\[3mm]
    \text{Length 5 subwords: }  \{ \langle \text{embe}, \text{embed}, \text{mbedd}, \text{beddi}, \text{eddin}, \text{dding}, \text{ding} \rangle \} 
    \\[3mm]
    \text{Length 6 subwords: }  \{ \langle \text{embed}, \text{embed}, \text{mbeddi}, \text{beddin}, \text{edding}, \text{dding} \rangle \} 
    \\[3mm]
    \text{And finally -- "<embedding>"}
    ```


Assume $\mathcal{G}_w$ is the set of all embeddings for a word, including the special subword. $g_i$ is the $i$th subword $\in \mathcal{G}_w$. 

For each input subword, FastText uses a hashing table to extract individual embedding vectors from the embedding matrix, $E$, that correspond with an index extracted by a hashing function, where the input is the subword embedding, $g_i$.

```math

g_i = \text{"<em"}\\[3mm]

\text{index} = \text{hash}(\text{"<em"}) \mod{|V|}
\\[3mm]
\text{index} = 12321 \mod{1000} = 321
\\[3mm]
e_{<em} = E_{321}

```

where $e_{<em}$ is the input to the model, which then takes the form of a word2vec type embedding model (make sure to use `PYTHONHASHSEED=X python myscript.py`). 

Once we extract all $e_j$ corresponding $g_i$, we'll end up with multiple $e_j$ for a single word $w \in |V|$. If we have the word "embedding" as the input, we have, ['<em', 'emb', 'mbe', 'bed', 'edd', 'ddi', 'din', 'ing', 'ng>'] and the special subword "$\text{<}$where$>$" -- resulting in a set, $\{e_1, e_2, \dots, e_{10}\} = \mathcal{E}_g$. 

We can simply sum over all $e_j$ in the set, $\sum_i^{10} e_i$, to form our final input vector for a single word -- $x_i$

> *Of course, for skipgram, you can immediately input this vector but for CBOW, we have to extract multiple $e_i$ and apply another operation as the average of all extracted $e_i$, $\frac{1}{\text{seq len}}\sum_i^{\text{seq len}} e_i$.*

There's always the possibility that there's a collision in the modulo operation, where multiple values of the $\text{hash}$ function yield a $0$ remainder when divided with $|V|$. 

FastText allows for this by reusing the same embedding vector for multiple subwords -- despite this the algorithm still remians efficient.

I conjecture another possible way to construct without collision could be as follows:

Each word $w$ is represented as a set of character n-grams (subwords) $\mathcal{G}_w$ of lengths between 3 and 6, plus a special subword for the word itself. To create the word's representation, a binary vector $\mathbf{s} \in \mathbb{R}^{|V|}$ where $s_i \in \{0, 1\}$ is constructed, where $V$ is the size of the subword vocabulary, and each index corresponds to a specific subword in $\mathcal{G}_w$. The binary vector $\mathbf{s}$ has a 1 at positions corresponding to the subwords present in the word input and 0 elsewhere. This binary vector is used to select subword embeddings $\mathbf{e}_{s_i}$ from the subword embedding matrix $\mathbf{E}_\text{subwords}$, as a matrix multiplication and the word embedding $\mathbf{e}_w$ is computed as the sum of the selected subword embeddings:

$$
\mathbf{e}_w = \sum_{s_i \in \mathcal{G}_w} \mathbf{e}_{s_i}
$$

> This can be easily generalized to sequence length $> 1$ and batch size $> 1$ if you take some time to derive it.

This method enables the model to handle out-of-vocabulary words by decomposing them into subwords and learning embeddings for both subwords and words.

> Though this'd probably require more computation.

### byte pair encoding

The issue with fastText can come when defining $n \in [a, b]$, such that vocabulary $|V|$ can be arbitrary -- and typically unreasonably large.



- [ ] figure out how byte pair encoding works -- break it down from first principles

---


<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>
<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>
