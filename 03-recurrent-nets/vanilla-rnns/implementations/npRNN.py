# two layer RNN, multiple time step (sequence) inputs - no backprop. i have pytorch autograd for that. 
# checkout my writeup on BPTT if you're interested in the maths behind it though!

# h = hidden size for 1st layer.

# Input X_t \in \mathbb{R}^(n \times d)
# First Layer Weights W^(1) \in \mathbb{R}^(d \times h)
# First Layer Hidden State Weights W_h^(1) \in \mathbb{R}^(h \times h)
# First Layer Bias b \in \mathbb{R}^(1 \times h)
# First layer Output, H_t^(1) \in \mathbb{R}^(n \times h)
# First Layer Hidden State, H_(t-1)^(1) \in \mathbb{R}^(n \times h)

# H_t = \phi(X_tW^(1) + W_h^(1)H_(t-1)^(1) + b)

# h_2 = hidden size for 2nd layer.

# Input H_t^(1) \in \mathbb{R}^(n \times h)
# Second Layer Weights W^(2) \in \mathbb{R}^(h \times h_2)
# Second Layer Hidden State Weights W_h^(2) \in \mathbb{R}^(h_2 \times h_2)
# Second Layer Bias b \in \mathbb{R}^(1 \times h_2)
# Second layer Output, H_t^(2) \in \mathbb{R}^(n \times h_2)
# Second Layer Hidden State, H_(t-1)^(2) \in \mathbb{R}^(n \times h_2)

# second layer
# H_t^(2) = \phi(H_t^(1)W^(2) + H_(t-1)^(2) W_h^(2) + b)


import numpy as np

class RNN():


    '''
    
    Notes:
    
    (i = idxs + 1) of n_layer, n_neurons, and activation_funcs determine which ith layer each 
    ith element of the respective tuple will be used to initialize the model.

    activation_funcs is a tuple of strings, either "tanh", "sigmoid", "relu", or "leaky relu"
    in_dim is the dimension of the embedding space for the inputs.
    batch_size is defaulted as 1 and only works with batch_size = 1 (i think, i haven't tested out with batch size > 1)
   
    '''
    
    def __init__(
        self,
        n_neurons: tuple,
        activation_funcs: tuple,
        in_dim: int,
        seq_len: int,
        batch_size = 1,
        ):
    
        np.random.seed(0)
        
        self.n_layers =  3
        self.n_neurons = n_neurons
        self.activation_funcs = activation_funcs
        self.in_dim = in_dim
        self.seq_len = seq_len
        self.batch_size = batch_size

        # init weights & internal outputs
        # total time steps = seq_len 
        self.weight, self.h_weight, self.bias = self._init_params()
        
        self.a = [
            [np.zeros(shape = (self.batch_size, n_neuron)) for _ in range(self.seq_len)] # for each time step
            for n_neuron in self.n_neurons # for each layer
        ]
           
        self.z = [
            [np.zeros(shape = (self.batch_size, n_neuron)) for _ in range(self.seq_len)] 
            for n_neuron in self.n_neurons
        ]
        
    def _init_params(self):
      
        weight = [0 for _ in range(self.n_layers)]
        h_weight = [0 for _ in range(self.n_layers)]
        bias = [0 for _ in range(self.n_layers)]
       
        for layer in range(self.n_layers):
           
            if layer == 0:
            
                weight[layer] = np.random.randn(
                    self.in_dim, 
                    self.n_neurons[layer] 
                )
               
                h_weight[layer] = np.random.randn(
                    self.n_neurons[layer],
                    self.n_neurons[layer]
                )
                
                bias[layer] = np.zeros((1, self.n_neurons[layer]))
                
            elif layer == 1:
                
                weight[layer] = np.random.randn(
                   self.n_neurons[layer - 1], 
                   self.n_neurons[layer] 
                )
                
                h_weight[layer] = np.random.randn(
                    self.n_neurons[layer],
                    self.n_neurons[layer] 
                )
      
                bias[layer] = np.zeros((1, self.n_neurons[layer]))

            elif layer == self.n_layers:
                
                weight[layer] = np.random.randn(
                    self.n_neurons[layer - 1],
                    self.n_neurons[layer] 
                )
                
                bias[layer] = np.zeros((1 , self.n_neurons[layer])) 

        return weight, h_weight, bias 

    def _forward(self, X):

        for t in range(self.seq_len): # tok_emb is the embedding for the ith token in the entire sequence , X
           
            tok_emb = X[:, :, t] 
            
            for layer in range(self.n_layers):
                if layer == 0:
                    self.z[layer][t] = np.dot(tok_emb, self.weight[layer]) + np.dot(self.a[layer][t], self.h_weight[layer]) + self.bias[layer]
                    self.a[layer][t] = self.activation(self.z[layer][t], self.activation_funcs[layer])
                elif layer == 1:
                    self.z[layer][t] = np.dot(self.a[layer - 1][t], self.weight[layer]) + np.dot(self.a[layer][t], self.h_weight[layer]) + self.bias[layer]
                    self.a[layer][t] = self.activation(self.z[layer][t], self.activation_funcs[layer])

        return self.a[-1][-1]
            
    def activation(self, z, activ):
        
        if activ == 'tanh':
            return np.tanh(z)
        elif activ == 'sigmoid':
            return 1 / (1 + np.exp( - z))
        elif activ == 'relu':
            return np.maximum(0, z)
        elif activ == 'leaky relu':
            return np.where(z > 0, z, z * .01)
        elif activ == 'softmax':
            return np.exp(z) / np.sum(np.exp(z), axis = 1, keepsdims = True) 
