# this is for bigram modelling.

import numpy as np
from sklearn.preprocessing import LabelEncoder

def generate_context_target_pairs(sentence, context_window=1):
    words_in_sentence = sentence.split()
    pairs = []
    for idx, target in enumerate(words_in_sentence):
        context = []
        for offset in range(-context_window, context_window + 1):
            context_idx = idx + offset
            if context_idx >= 0 and context_idx < len(words_in_sentence) and context_idx != idx:
                context.append(words_in_sentence[context_idx])
        for c in context:
            pairs.append(([vocab[c] for c in context], vocab[target]))
    return pairs


sentences = [
    "The cat sat on the mat",
    "The dog chased the cat",
    "The cat chased the mouse",
    "The mouse ran under the table",
    "The dog barked loudly"
]

words = set(" ".join(sentences).split())
vocab = {word: idx for idx, word in enumerate(words)}

print(f"Vocab: {vocab}\n")

context_target_pairs = []
for sentence in sentences:
    context_target_pairs.extend(generate_context_target_pairs(sentence))

print(f"Context Target Pairs: {context_target_pairs}")
