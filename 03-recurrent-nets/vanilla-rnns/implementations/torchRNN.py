import torch
from torch import nn

class RNN(nn.Module):
    
    def __init__(self):
        
        super().__init__()
      
        self.embedding = nn.Embedding(
            num_embeddings = 27,
            embedding_dim = 20 
            )  
        
        self.rnn1 = nn.RNN(
            input_size = 20, 
            hidden_size = 32
            )
        
        self.rnn2 = nn.RNN(
            input_size = 32,  
            hidden_size = 64
            )
        
        self.rnn3 = nn.RNN(
            input_size = 64 ,
            hidden_size = 64
        )
        
        self.fc4 = nn.Linear(
            in_features = 64,
            out_features = 26
        )
        
    def forward(self, x):
        
        x = self.embedding(x)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x, _ = self.rnn3(x)
        
        
        x = self.fc4(x)
        return x        

