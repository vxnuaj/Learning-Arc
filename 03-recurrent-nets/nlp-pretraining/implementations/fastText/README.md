# training word2vec with subword embeddings


As the title suggests, I'll be training word2vec, with subword embeddings using the fastText algorithm.

I'll be using the text8 dataset -- luckily for me it's pretty much cleanesed and preprocessed, so I don't need to do much heavy lifting there (i dislike preprocessing data).

for a quick description:

> "*The Text8 dataset is a large corpus of text used for training and evaluating language models in natural language processing (NLP). Derived from Wikipedia articles, it contains approximately 100 million tokens and has a vocabulary size of around 253,855 unique words or subwords. The text has been preprocessed to remove punctuation, convert all text to lowercase, and replace numbers with a special token. The dataset is commonly used for tasks such as language modeling, text generation, and text classification, and is particularly useful for training recurrent neural networks (RNNs) and transformers. It is publicly available and can be downloaded from various sources, making it a popular choice for researchers and developers in the field of NLP.*"
>
> ***courtesy of llama-3.3-70b***