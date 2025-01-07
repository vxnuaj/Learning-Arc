## Working w Sequences

- Previously it was assumed all inputs to a given neural network, $\mathcal{N}$, were sampled independently from a distribution $P(X)$.
- Some datasets consist of massive sequences
- For an single sample sequence, $S_i$, we can stil assume $S_i$ is independent from $S_{i±n \geq 0}$, where $n$ is any integer.
- But assuming $S$ is a sequence with multiple datapoints, each occuring at different timesteps $t$, we can't assume that all datapoints, $s_t$ are independent of each other.
  - words at the end of a sentence are dependent on words at the beginning of a sentence.
    - this allows us to reliably model language as it's clear there's a (semantic and syntactic) relationship between words in a sentence.
- Sequence models don't require stationarity ($P$ remaining the same over $t$ or position $p$, for instance)

### Targets

Sometimes, language models aim to predict a fixed target $y$, given a sequential input. We take in a sequence of data which changes over time $t$ and output a single variable.

Other times we aim to predict a sequence, $Y = y_1, y_2, \dots, y_T$, to a fixed input. We take in a single input which does not change over $t$, and we feed the entire input to the model.

In the former, the sequence is fed into a model with multiple time-steps $t$ while in the latter the entire sequence is fed to the model at once.

Other times, we want to predict sequentially structured outputs based on sequentialy structure inputs (Seq2Seq tasks, such as machine translation or video captioning)

These take 2 forms:

- *Aligned*, where each input at each time step corresponds with a target output
- *Unaligned* where the input at each time step doesn't neccesarily have an output.

**Unsupervised (well self-supervised) Density Modelling (sequence modelling):** 

Where given a collection of sequences, the goal is to estimate the probability mass function that tells us how likely we are to see any given sequence, $p(x_1, \dots, x_t)$

For a given collection of sequence $X$, the model learns to model the probability of a given sequence $x_t$ appearing in the dataset.

### Tokenization

A given sequence is divided into tokens, which can be words, subwords, or characrters.

For sequence modelling, a single input correpsonds to a single token (typically).

### Pre-processing

To preprocess a text dataset, we generally follow these steps:


1. Split text into tokens (characters, words, or subwords)
2. Build a vocabulary of your tokens
3. Convert the voabulary into numerical indices where each word has a unique index.
4. Construct input sequences (or input datas in general)
5. Convert text data (input sequences) into numerical sequences based on the numerical vocabulary.

other steps prior include removal of stopwrods, punctuation, lowercasing, etc.