import sentencepiece as spm

with open('/Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/LABS/DeepLearning/03-recurrent-nets/nlp-pretraining/implementations/data/enwiki8/text8', 'r') as f:
    
    text8 = f.read()

print('stripping and splitting words')

text8 = text8.strip() 
text8 = text8.split()

print(f"word count: {len(text8)}")

vocab = set(text8)

print('joining words')

vocab_str = ' \n'.join(vocab) 
 
print('writing into vocab.txt') 
  
with open('/Users/juanvera/Documents/DEVELOPMENT/SINGULARITY/LABS/DeepLearning/03-recurrent-nets/nlp-pretraining/implementations/fastText/vocab.txt ', 'w') as f:
    f.write(vocab_str + ' \n')
   
'''
spm.SentencePieceTrainer.train(
    
    input = text8, 
    model_prefix = 'txt8',
    
    )
'''

