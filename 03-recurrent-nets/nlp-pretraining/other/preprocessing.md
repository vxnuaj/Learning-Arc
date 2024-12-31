### NLP Preprocessing

**Tokenization**:

- Breaks down text into smaller units, such as individual words, subwords, or sentences
- Needs to be done, to generate token embeddings 

**Text Normalization**
- Lower case all words.
- Remove punctuation, unless needed for a given task -- remove anything that doesn't contribute to semantic meaning of the text.
- Remove special characters that might not contribute to analysis -- emoji's, html tags, etc.

**Stopword Removal**
- Remove words that don't ahve significant meaning, such as 'the', 'is', 'in', etc. to the given task.

**Stemming and Lemmatization**
- Stemmign: Reduce words by chopping prefixes / suffixes (running $\rightarrow$ run)
- Lemmatization: Reduce words to simpler 'lemma' -- their most basic form --, which satisfy the root meaning.


