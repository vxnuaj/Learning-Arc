import numpy as np
from neuralops import ConvOps as cops
from neuralops import FCOps as fcops
from functions import Functions as F
from utils import validate_inputs
from utils import Transforms as T
from utils import Encoding as E

class ConvNet:
    
    def __init__(self, seed:int = None):
        
        '''
        Seed: Set the seed for reproducibility
        ''' 
        
        self.seed = seed
        self._set_seed()
       
    def train(
        
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        layers: list[str], 
        n_output_channels: list[int], 
        layer_size: list[tuple[int, int] | int], 
        activations: list[str],
        y_onehot = None,
        padding: list[tuple[int, int] | int] = 0, 
        stride: list[tuple[int, int] | int] = 1, 
        dilation_rate: list[tuple[int, int] | int] = 1,
        param_init = 'glorot',
        alpha = .01,
        epochs = 100
    ):
        
        '''
        X: Input Data, of shape (B, C, H, W) 
        layers: a list of strings, where each string is either 'C', 'MP', 'AP', 'FC', where 'C' denotes a convolutional layer, 'MP' denotes as max pooling layer, 'AP' denotes an average pooling layer, and 'FC' denotes a fully connected layer. 
        '''


        self.layers = layers
        
        ###### DEFINING LAYER COUNT, FOR CONVOLUTIONAL BLOCK, CONVOLUTIONS, AND FC LAYERS ######
         
        self._n_layers = len(layers) 
        self._n_conv_block_layers = len([i for i in self.layers if i in ['C', 'MP', 'AP']])
        self._n_conv_layers = len([i for i in self.layers if i == 'C'])
        self._n_fc_layers = len([i for i in self.layers if i == 'FC'])
            
        ###### VALIDIATING INPUTS VIA ASSERTIONS (utils.validate_inputs) ###### 
        
        validate_inputs(
          
            X = X,
            y = y,
            layers = layers,
            output_channels = n_output_channels,
            layer_size = layer_size,
            activations = activations,
            y_onehot = y_onehot,
            padding = padding,
            stride = stride,
            dilation_rate = dilation_rate,
            param_init = param_init,
            n_layers = self._n_layers,
            n_conv_layers = self._n_conv_layers,
            n_conv_block_layers = self._n_conv_block_layers,
            n_fc_layers = self._n_fc_layers,
            alpha = alpha,
            epochs = epochs
            
            )       
        

        self.padding = padding
        self.X = T.pad(X, self.padding[0])
        self.y = y.reshape(1, -1)
        self.n_output_channels = n_output_channels
        self.layer_size = layer_size
        self.activations = activations
        self.y_onehot = y_onehot
        self.stride = stride
        self.dilation_rate = dilation_rate 
        self.param_init = param_init
        self.alpha = alpha
        self.epochs = epochs

        ###### INITILAIZING PARAMS | CONV KERNELS AND FC WEIGHTS ###### 
                
        self.weights, self.bias = self._init_params()

        ###### GRADIENT DESCENT ######

        self._gradient_descent()

        return self.weights, self.bias

    def test(self, X_test, y_test, y_onehot_test = None):
       
        self.X_test = T.pad(X_test, self.padding[0])
        self.y_test = y_test.reshape(1, -1)
        self.y_onehot_test = y_onehot_test
       
        prob = self._test() 
        pred = np.argmax(prob, axis = 0)
        loss = F.CrossEntropyLoss(self._y_onehot_test, prob)
        
        accuracy = F.Accuracy(self.y_test, pred)  
      
        print(f"Testing Loss: {loss} | Testing Accuracy: {accuracy}%") 
        
        return

    def _gradient_descent(self):
      
        for epoch in range(self.epochs):
            
            prob = self._forward()
            pred = np.argmax(prob, axis = 0)
            loss = F.CrossEntropyLoss(self._y_onehot, prob)
            
            accuracy = F.Accuracy(self.y, pred)
           
            print(f"Epoch: {epoch + 1} | Loss: {loss} | Accuracy: {accuracy}%")

            self._backward()
            self._update()

    def _test(self): ### FORWARD PASS
        
        batch_size = self.X_test.shape[0]
        self.__a_test = [0 for _ in range(self._n_layers)]
        self.__z_test = [0 for _ in range(self._n_layers)]
     
        for l in range(self._n_layers):
          
           
            if self.layers[l] == 'C': ### IF WE ARE AT A CONV LAYER
                
                kernel = self.weights[l]
                b = self.bias[l]
                stride = self.stride[l] 

                if isinstance(self.activations, list):
                    act_func = F.get_func(self.activations[l])
                
                else:
                    act_func = F.get_func(self.activations)
                
                if l == 0:
                    self.__z_test[l] = cops.conv2D(self.X_test, kernel, stride) + b
                    self.__a_test[l] = act_func(self.__z_test[l])
                
                else:
                    self.__z_test[l] = cops.conv2D(self.__a_test[l - 1], kernel, stride) + b
                    self.__a_test[l] = act_func(self.__z_test[l])
      
            elif self.layers[l] == 'MP': ### IF WE'RE AT A MAX POOLING LAYER
                
                kernel = self.weights[l] 
                stride = self.stride[l] 
               
                if l == 0:
                   
                    self.__a_test[l], self.__mp_indices[l] = cops.maxpool2D(self.X_test, kernel, stride, self.n_output_channels[l], return_idxs = True )
                    self.__mp_indices[l] = self.__mp_indices[l].astype(int)

                else: 
                   
                    self.__a_test[l], self.__mp_indices[l] = cops.maxpool2D(self.__a_test[l - 1], kernel, stride, self.n_output_channels[l], return_idxs = True) 
                    self.__mp_indices[l] = self.__mp_indices[l].astype(int)
      
            elif self.layers[l] == 'AP': ### IF WE'RE AT AN AVERAGE POOLING LAYER
              
                kernel = self.weights[l]
                stride = self.stride[l]
                
                if l == 0:
                    self.__a_test[l] = cops.avgpool2D(self.X_test, kernel, stride, self.n_output_channels[l]) 

                else:
                    self.__a_test[l] = cops.avgpool2D(self.__a_test[l - 1], kernel, stride, self.n_output_channels[l])

            elif self.layers[l] == 'FC': # IF WE'RE AT A FULLY CONNECTED LAYER
                
                w = self.weights[l]
                b = self.bias[l]

                if isinstance(self.activations, list):
                    
                    act_func = F.get_func(self.activations[l])
                    
                else:
                    
                    act_func = F.get_func(self.activations)

                if self.layers[l - 1] in ['C', 'AP', 'MP']:
                   
                    a = self.__a_test[l - 1].reshape(batch_size, -1).T
                  
                    self.__z_test[l] = np.dot(w, a) + b
                    self.__a_test[l] = act_func(self.__z_test[l])
                  
                else:
                    
                    self.__z_test[l] = np.dot(w, self.__a_test[l-1])
                    self.__a_test[l] = act_func(self.__z_test[l]) 

        return self.__a_test[-1]

    def _forward(self): ### FORWARD PASS
        
        batch_size = self.X.shape[0]
        self.__a = [0 for _ in range(self._n_layers)]
        self.__z = [0 for _ in range(self._n_layers)]
        self.__mp_indices = [0 for _ in range(self._n_layers)] 
       
        for l in range(self._n_layers):
           
            if self.layers[l] == 'C': ### IF WE ARE AT A CONV LAYER
                
                kernel = self.weights[l]
                b = self.bias[l]
                stride = self.stride[l] 

                if isinstance(self.activations, list):
                    act_func = F.get_func(self.activations[l])
                
                else:
                    act_func = F.get_func(self.activations)
                
                if l == 0:
                    self.__z[l] = cops.conv2D(self.X, kernel, stride) + b
                    self.__a[l] = act_func(self.__z[l])
                   
                else:
                    self.__z[l] = cops.conv2D(self.__a[l - 1], kernel, stride) + b
                    self.__a[l] = act_func(self.__z[l])
                  
      
            elif self.layers[l] == 'MP': ### IF WE'RE AT A MAX POOLING LAYER
                
                kernel = self.weights[l] 
                stride = self.stride[l] 
               
                if l == 0:
                   
                    # TODO
                    
                    self.__a[l], self.__mp_indices[l] = cops.maxpool2D(self.X, kernel, stride, self.n_output_channels[l], return_idxs = True )
                    self.__mp_indices[l] = self.__mp_indices[l].astype(int)

                else: 
                   
                    self.__a[l], self.__mp_indices[l] = cops.maxpool2D(self.__a[l - 1], kernel, stride, self.n_output_channels[l], return_idxs = True) 
                    self.__mp_indices[l] = self.__mp_indices[l].astype(int)
      
            elif self.layers[l] == 'AP': ### IF WE'RE AT AN AVERAGE POOLING LAYER
              
                kernel = self.weights[l]
                stride = self.stride[l]
                
                if l == 0:
                    self.__a[l] = cops.avgpool2D(self.X, kernel, stride, self.n_output_channels[l]) 

                else:
                    self.__a[l] = cops.avgpool2D(self.__a[l - 1], kernel, stride, self.n_output_channels[l])

            elif self.layers[l] == 'FC': # IF WE'RE AT A FULLY CONNECTED LAYER
                
                w = self.weights[l]
                b = self.bias[l]

                if isinstance(self.activations, list):
                    
                    act_func = F.get_func(self.activations[l])
                    
                else:
                    
                    act_func = F.get_func(self.activations)

                if self.layers[l - 1] in ['C', 'AP', 'MP']:
                    
                    a = self.__a[l - 1].reshape(batch_size, -1).T
                    self.__z[l] = np.dot(w, a) + b
                    self.__a[l] = act_func(self.__z[l])
                  
                else:
                    
                    self.__z[l] = np.dot(w, self.__a[l-1])
                    self.__a[l] = act_func(self.__z[l]) 

  
        return self.__a[-1]
        
    def _backward(self): ### BACKPROPAGATING GRADIENTS VIA CHAIN RULE

        self.__dz = [np.zeros_like(self.__a[l]) if l in ['FC', 'C'] else 0 for l, _ in enumerate(self.layers)]
        self.__dw = [np.zeros_like(self.__a[l]) if l in ['FC', 'C'] else 0 for l, _ in enumerate(self.layers)]
        self.__db = [np.zeros_like(self.__a[l]) if l in ['FC', 'C'] else 0 for l, _ in enumerate(self.layers)]
     
        batch_size = self.X.shape[0] 
      
        for l, _ in reversed(list(enumerate(self.layers))):
            
            act_func = F.get_func(self.activations[l]) 

            if self.layers[l] == 'FC': ### IF WE'RE AT A FULLY CONNECTED LAYER
                
                if l == (self._n_layers - 1): # IF WE'RE AT THE LAST LAYER
                    
                    self.__dz[l], self.__dw[l], self.__db[l] = fcops.grad_last_layer(self.y_onehot, self.__a, batch_size, l)
                    
                elif self.layers[l - 1] in ['MP', 'AP', 'C']: # IF THE FC LAYER IS RIGHT AFTER THE CONVOLUTIONAL BLOCK / THE CONVOLUTIONAL BLOCK IS RIGHT BEFORE THE GIVEN FC LAYER
                    
                    self.__dz[l], self.__dw[l], self.__db[l] = fcops.grad_cblock_prior(self.__a, self.weights, self.__z, batch_size, act_func, l)
                    
                else: # IF THE FC LAYER IS IN THE MIDDLE OF THE FC BLOCK    
                    
                    self.__dz[l], self.__dw[l], self.__db[l] = fcops.grad_fcblock(self.__a, self.weights, self.__z, batch_size, act_func, l) 
                  
            elif self.layers[l] == 'C':
                
                if self.layers[l + 1] == 'FC': # IF THE LAYER AFTER THE CONVOLUTION LAYER IS AN FC LAYER
              
                    if l - 1 < 0: 
                        self.__dz[l], self.__dw[l], self.__db[l] = cops.grad_convblock_from_fc(self.weights[l + 1], z_1 = self.__z[l], z_2 = self.__z[l+1], a_0 = self.X, kernel = self.weights[l], stride = self.stride[l], act_func=act_func)
                    else:
                        self.__dz[l], self.__dw[l], self.__db[l] = cops.grad_convblock_from_fc(self.weights[l + 1], z_1 = self.__z[l], z_2 = self.__z[l+1], a_0 = self.__a[l - 1], kernel = self.weights[l], stride = self.stride[l], act_func=act_func)
                      
                elif self.layers[l + 1] == 'C':
               
                    if l - 1 < 0:
                        
                        self.__dz[l], self.__dw[l], self.__db[l] = cops.grad_conv2D(self.__dz[l+1], self.__z[l], self.X, self.weights[l], self.weights[l + 1], self.stride[l + 1], self.n_output_channels[l], act_func, out_shape = self.__z[l].shape)
                      
                    else:
                        self.__dz[l], self.__dw[l], self.__db[l] = cops.grad_conv2D(self.__dz[l+1], self.__z[l], self.__a[l-1], self.weights[l], self.weights[l+1], self.stride[l+1], self.n_output_channels[l], act_func, out_shape = self.__z[l].shape)
                  
                elif self.layers[l+ 1] == 'MP':
              
                    if l - 1 < 0:
                        
                        self.__dz[l], self.__dw[l], self.__db[l] = cops.grad_conv2D_from_maxpool(self.__dz[l+1], self.__z[l], self.X, self.__mp_indices[l+1], act_func, self.stride[l], self.weights[l])

                    else:
                        self.__dz[l], self.__dw[l], self.__db[l] = cops.grad_conv2D_from_maxpool(self.__dz[l+1], self.__z[l], self.__a[l - 1], self.__mp_indices[l + 1], act_func, self.stride[l], self.weights[l])
                    
                elif self.layers[l + 1] == 'AP':
             
                    if l - 1 < 0:

                        self.__dz[l], self.__dw[l], self.__db[l] = cops.grad_conv2D_from_avgpool(self.__dz[l+1], self.__z[l], self.X, self.weights[l], self.stride[l], act_func)  
           
                    else:
                     
                        self.__dz[l], self.__dw[l], self.__db[l] = cops.grad_conv2D_from_avgpool(self.__dz[l+1], self.__z[l], self.__a[l-1], self.weights[l], self.stride[l], act_func) 
               
            elif self.layers[l] == 'MP':
                
                if self.layers[l + 1] == 'FC': 
                   
                    self.__dz[l] = cops.grad_convblock_from_fc(self.weights[l + 1], z_2 = self.__z[l + 1], a_1 = self.__a[l], layer = 'MP', act_func = act_func)

                elif self.layers[l + 1] == 'C': # if the layer after the maxpool layer was the conv layer.
                    
                    '''
                  
                     IMPLEMENTED W GUIDANCE FROM THE LINK BELOW 
                   
                    https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html: 
                    
                    "The padding argument effectively adds dilation * (kernel_size - 1) - padding amount of zero padding to both sizes of the input. 
                    This is set so that when a Conv2d and a ConvTranspose2d are initialized with same parameters, they are inverses of each other in 
                    regard to the input and output shapes. However, when stride > 1, Conv2d maps multiple input shapes to the same output shape. 
                    output_padding is provided to resolve this ambiguity by effectively increasing the calculated output shape on one side. 
                    Note that output_padding is only used to find output shape, but does not actually add zero-padding to output."

                    ''' 

                    self.__dz[l] = cops.grad_maxpool2D(self.__dz[l+1], self.weights[l+1], self.stride[l+1], self.n_output_channels[l], self.__a[l].shape) 
                  
            elif self.layers[l] == 'AP':
                
                '''
               
                "You can compute the gradients of a global avg pool w.r.t the inputs - it's simply dividing by the number of elements pooled. 
              
                https://stackoverflow.com/questions/70691539/how-does-error-get-back-propagated-through-pooling-layers               
                https://stats.stackexchange.com/questions/565032/cnn-upsampling-backprop-gradients-across-average-pooling-layer  
               
                '''
                
                if self.layers[l + 1] == 'FC':
                 
                    self.__dz[l] = cops.grad_convblock_from_fc(self.weights[l + 1], z_2 = self.__z[l + 1], a_1 = self.__a[l], layer = 'AP', act_func = act_func)  
                   
                elif self.layers[l + 1] == 'C': 
                    
                    self.__dz[l] = cops.grad_avgpool2D(self.__dz[l+1], self.weights[l+1], self.stride[l+1], self.n_output_channels[l], self.__a[l].shape)
        
        return
   
    def _update(self):
       
        # TODO
       
        for l in range(self._n_layers):
           
            if self.layers[l] in ['C', 'FC']:
                
                self.weights[l] -= self.alpha * self.__dw[l]
                self.bias[l] -= self.alpha * self.__db[l]
           
            elif self.layers[l] in ['MP', 'AP']:
           
                pass  # no params in max pooling or average pooling layers
            
        return
    
    def _init_params(self):
        
        weights = []
        bias = [] 
        conv_out_dims = [] 
        
        for l, l_type in enumerate(self.layers):
            if l_type.upper() == 'C':
                _bool = self._check_dilation(l) # True if dilation is set for the lth conv layer, else is False
                
                if not _bool: # Default kernel creation if dilation is not expected
                   
                    weights.append(self._init_kernel(l))
                
                elif _bool: # Dilated kernel if dilation is expected
                    
                    kernel_mask = self._init_kernel(l)
                    weights.append(T.dilate_kernel(kernel_mask, self.dilation_rate[l]))
                
                bias.append(np.zeros(shape = (self.n_output_channels[l], 1, 1)))
              
                if l == 0:  
                   
                    conv_out_dims.append(self._out_feat_map_size(
                            input_size = self.X.shape[2:4], 
                            kernel_size = weights[l].shape[2:4],
                            padding = self.padding[l],
                            stride = self.stride[l] 
                        ))
                    
                else:
                    
                    conv_out_dims.append(self._out_feat_map_size(
                            input_size = conv_out_dims[l - 1],
                            kernel_size = weights[l].shape[2:4],
                            padding = self.padding[l],
                            stride = self.stride[l] 
                        ))
        
            elif l_type.upper() in ['MP', 'AP']:
                
                weights.append(self.layer_size[l]) 
                bias.append(float('nan')) 
              
                if l == 0:
                   
                    conv_out_dims.append(self._out_feat_map_size(
                        input_size = self.X.shape[2:4],
                        kernel_size = weights[l],
                        padding = self.padding[l],
                        stride = self.stride[l] 
                    ))
               
                else:
                    
                    conv_out_dims.append(self._out_feat_map_size(
                        input_size = conv_out_dims[l-1],
                        kernel_size = weights[l],
                        padding = self.padding[l],
                        stride = self.stride[l]
                    ))
          
            elif l_type.upper() == 'FC':
                weights.append(self._init_fc(l, conv_out_dims))
                bias.append(np.zeros(shape = (self.layer_size[l], 1))) 
      
        return weights, bias   
    
    def _init_kernel(self, l):
        
         
        if l == 0:
            n_in = self.X.shape[1] * self.layer_size[l][0] * self.layer_size[l][1]
        else:
            n_in = self.n_output_channels[l - 1] * self.layer_size[l][0] * self.layer_size[l][1] 
        
        if self.param_init == 'glorot': 
            if l == 0:
                kernel = np.random.randn(self.n_output_channels[l], self.X.shape[1], *self.layer_size[l]) * np.sqrt(1 / (n_in))
            else:
                kernel = np.random.randn(self.n_output_channels[l], self.n_output_channels[l - 1], *self.layer_size[l]) * np.sqrt(1 / (n_in))
        elif self.param_init == 'kaiming': 
            if l == 0:
                kernel = np.random.randn(self.n_output_channels[l], self.X.shape[1], *self.layer_size[l]) * np.sqrt(2 / (n_in))
            else:
                kernel = np.random.randn(self.n_output_channels[l], self.n_output_channels[l - 1], *self.layer_size[l]) * np.sqrt(2 / (n_in))

        return kernel
 
    def _init_fc(self, l, conv_out_dims):

        assert isinstance(self.layer_size[l], int), ValueError(f'index {l} layer_size corresponds to a fc layer. params cannot be initialized as {type(self.layer_size)}, must be type int')
      
        if self.param_init == 'glorot':
            if self.layers[l - 1].upper() in ['C', 'MP', 'AP']:
                
                h, w = conv_out_dims[l - 1] 
                conv_out_size = self.n_output_channels[l - 1] * h * w
               
                weights = np.random.randn(self.layer_size[l], conv_out_size ) * np.sqrt(1/(conv_out_size))            
                
            else:
                weights = np.random.randn(self.layer_size[l], self.layer_size[l - 1]) * np.sqrt(1 / (self.layer_size[l - 1]))
                
        elif self.param_init == 'kaiming':
            if self.layers[l - 1].upper() == 'C':
                
                h, w = conv_out_dims[l - 1] 
                conv_out_size = self.n_output_channels[l - 1] * h * w                 
                weights = np.random.randn(self.layer_size[l], conv_out_size) * np.sqrt(2 / (conv_out_size)) 
                
            else:
                weights = np.random.randn(self.layer_size[l], self.layer_size[l - 1]) * np.sqrt (2 / self.layer_size[l - 1])       

        return weights
  
    def _check_dilation(self, idx):
       
        if self.dilation_rate[idx] == 1 or self.dilation_rate[idx] == (1, 1): 
            return False
        elif isinstance(self.dilation_rate[idx], int) and self.dilation_rate[idx] > 1:
            return True
        elif isinstance(self.dilation_rate[idx], tuple) and (self.dilation_rate[idx][0] > 1 or self.dilation_rate[idx][1] > 1):
            return True
      
    def _out_feat_map_size(self, input_size, kernel_size, padding, stride) -> tuple[int, int]:
        
        '''
        Takes in a tuple for all arguments, each tuple containing hyperparams for (H, W) respectively.
       
        We don't use padding in the formula as input_size[0] and input_size[1] already denotes the padded input size to the lth layer.
        
        ''' 
        
        h = ((input_size[0] - kernel_size[0]) // stride[0]) + 1
        w = ((input_size[1] - kernel_size[1]) // stride[1]) + 1
       
        return (h, w)
       
    def _set_seed(self):
        if self.seed is not None:
            np.random.seed(self.seed)
    
    @property
    def seed(self):
        return self._seed
    
    @seed.setter
    def seed(self, seed):
        assert isinstance(seed, int), ValueError('seed must be type int.') 
        self._seed = seed
       
    @property
    def n_output_channels(self):
        return self._n_output_channels
    
    @n_output_channels.setter
    def n_output_channels(self, n_output_channels):
       
        assert len(n_output_channels) == self._n_conv_layers, ValueError("n_output_channels must be same length as the number of 'C' layers")
        
        for l in range(self._n_layers):
            
            if (self.layers[l] == 'MP' or self.layers[l] == 'AP') and self.layers[l - 1] == 'C':
                n_output_channels.insert(l, n_output_channels[l - 1]) 
            elif (self.layers[l] == 'MP' or self.layers[l] == 'AP') and  l == 0:
                n_output_channels.insert(l, self.X.shape[1])
        
        self._n_output_channels = n_output_channels
      
    @property
    def layer_size(self):
        return self._layer_size 
      
    @layer_size.setter
    def layer_size(self, layer_size):
     
        if not isinstance(layer_size, list):
            raise ValueError("layer_size must be a list.")

        for param in layer_size:
            if not (isinstance(param, int) or (isinstance(param, tuple) and len(param) == 2 and all(isinstance(i, int) for i in param))):
                raise ValueError("Each element of layer_size must be either an int or a tuple of two ints.")
       
        for i, n in enumerate(layer_size):
            if isinstance(i, int) and self.layers[i] != 'FC':
                layer_size[i] = (n, n)
        
                
        self._layer_size = layer_size
      
    @property
    def activations(self):
        return self._activations 
    
    @activations.setter
    def activations(self, activations):
       
        for i in range(self._n_layers):
            if self.layers[i] in ['AP', 'MP']:
                activations.insert(i, float('nan'))
        
        self._activations = activations
       
    @property
    def stride(self):
        return self._stride
    
    @stride.setter
    def stride(self, stride):
        if isinstance(stride, int):
            stride_seq = [stride for _ in range(self._n_conv_block_layers)]
            stride = stride_seq 
        for i, s in enumerate(stride):
            if isinstance(s, int):
                stride[i] = (s, s)
            else:
                pass
        self._stride = stride
           
    @property
    def padding(self):
        return self._padding
    
    @padding.setter
    def padding(self, padding):
        
        if isinstance(padding, int):
            pad_seq = [padding for _ in range(self._n_conv_block_layers)] 
            padding = pad_seq
        for i, p in enumerate(padding):
            if isinstance(p, int):
                padding[i] = (p, p)
            else:
                pass
        self._padding = padding 

    @property
    def dilation_rate(self):
        return self._dilation_rate
    
    @dilation_rate.setter
    def dilation_rate(self, dilation_rate):
       
        for i, lyr in enumerate(self.layers):
            if lyr in ['MP','AP']:
                dilation_rate.insert(i, (1, 1))
        
        assert len(dilation_rate) == self._n_conv_block_layers, 'Amount of dilation rates cannot be greater than the amount of convolutional layers.'
                
        self._dilation_rate = dilation_rate        

    @property
    def y_onehot(self):
        return self._y_onehot

    @y_onehot.setter
    def y_onehot(self, y_onehot):
        if y_onehot is None:
            y_onehot = E.one_hot_encode(self.y) 
        self._y_onehot = y_onehot
        
    @property
    def y_onehot_test(self):
        return self._y_onehot_test
    
    @y_onehot_test.setter
    def y_onehot_test(self, y_onehot_test):
        if y_onehot_test is None:
            y_onehot_test = E.one_hot_encode(self.y_test)
        self._y_onehot_test = y_onehot_test
