'''


'''

import numpy as np

class CBOW:

    '''
    
    '''
    
    def __init__(self, vocab_size, embedding_dim):
       
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim 
        
        pass
  
    def train(self, x, y):
        
        pass
   
    def _init_params(self, ):
       
        self.w1 = np.random.randn(
            self.embedding_dim,
            self.vocab_size
            )
        
        self.b1 = np.zeros(
            shape = (self.embedding_dim, 1)
            )
       
        self.w2 = np.random.randn(
            
            
            
        )
        
        self.b2 = np.zeros(
            
            
        )

        
        pass