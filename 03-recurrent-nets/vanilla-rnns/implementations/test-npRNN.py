import numpy as np
from npRNN import RNN

# init data + assume we have a vocabulary size of 50.
X = np.random.randn(1, 50, 5) # batch size = 1, embedding_dim = 50, seq_len = 5
one_hot_shape = (1, 50)

# init internals
h_units = (10, 20, 50) # idx = layer num
activation_funcs = ('tanh', 'tanh')
in_dim = 50
seq_len = 5
batch_size = 1

# init model and forward pass
rnn = RNN(h_units, activation_funcs, in_dim, seq_len, batch_size)
print(f"Output Shape: {rnn._forward(X).shape}") 
print(f"Right Output Shape? {one_hot_shape == rnn._forward(X).shape}")
