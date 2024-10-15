'''
TODO LEFT OFF HERE:
           
    - LEFT OFF ON LAYER 3, JUST COMPUTED THE GRADIENTS FOR LAYER 3 CORRECTLY! REST SHOULD BE SMOOTH CACLUALTIONS.
            
X: Input Data, of shape (B, C, H, W) 

- any other hidden feature maps are of same size, as (B, C, H, W)


Each are in their respective lists:
    Kernel: of shape (Out Channels, In Channels, Height, Width)
    Bias: of shape (Out Channels, ), as we have bias parameter per output channel for a given layer
    
If accounting for the list, the shape becomes:
    Kernel: (Layers, Out Channels, In Channels, Height, Width)
    Bias: (Layers, Out Channels)

MAIN TODO

- [ ] Build NumPy ConvNet
    - [X] Convolutional Layers
    - [X] Fully Connected Layers
    - [ ] Pooling Layers
    - [X] Forward Pass
    - [ ] Backward Pass
    - [ ] Gradient Descent
- [ ] Train on a small toy dataset

shoudl have ablity to define different padding, stride, dilation, etc for height and width of a conv

OTHER TODO

- [ ] Ensure that number of fc params for the first fc layers is equal to its flattened input (assertion)?
- [ ] Split the entire code up into different files?
- [ ] when shipped to git repo, define structure and utilization (add the run.py)
- [ ] different optimizers?
- [ ] perhaps change functionality of assertions such that they can work on raw args, and not the self.[args]

'''

import numpy as np
from neuralops import ConvOps as cops
from neuralops import FCOps as fcops
from functions import Functions as F
from utils import validate_inputs
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
        param_init = 'glorot' 
    ):
        
        '''
        X: Input Data, of shape (B, C, H, W) 
        layers: a list of strings, where each string is either 'C', 'MP', 'AP', 'FC', where 'C' denotes a convolutional layer, 'MP' denotes as max pooling layer, 'AP' denotes an average pooling layer, and 'FC' denotes a fully connected layer. 
        '''


        self.layers = layers
        
        # define layer count: total, conv, and fc 
        self._n_layers = len(layers) 
        self._n_conv_block_layers = len([i for i in self.layers if i in ['C', 'MP', 'AP']])
        self._n_conv_layers = len([i for i in self.layers if i == 'C'])
        self._n_fc_layers = len([i for i in self.layers if i == 'FC'])
       
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
            n_fc_layers = self._n_fc_layers
            
            )       
        
        self.padding = padding
        self.X = self._pad(X, self.padding[0])
        self.y = y.reshape(1, -1)
        self.n_output_channels = n_output_channels
        self.layer_size = layer_size
        self.activations = activations
        self.y_onehot = y_onehot
        self.stride = stride
        self.dilation_rate = dilation_rate 
        self.param_init = param_init

        
        self.weights, self.bias = self._init_params()

        '''
        print(f"weights at conv1\n\n{self.weights[0]}\n\n")
        print(f"weights at conv2\n\n{self.weights[1]}\n\n")
        print(f"weights at fc1\n\n{self.weights[2].shape}\n\n")
        print(f"weights at fc2\n\n{self.weights[3].shape}\n\n")
        
        print(f"Bias:\n\n{self.bias}\n")
        print(self.bias[0].shape)
        print(self.bias[1].shape)  
        
        '''
       
        logits = self._forward() 
        print(f"Logits shape: {logits.shape}")
        pred = np.argmax(logits, axis = 0)
        loss = F.CrossEntropyLoss(self.y, logits)
        accuracy = F.Accuracy(y, pred)
        print(f"Predictions: {pred}")
        print(f"Loss: {loss}")
        print(f"True Labels: {y}")
        print(f"Accuracy: {accuracy}%")
        self._backward()     

    def _forward(self):
        
        batch_size = self.X.shape[0]
        self.__a = [0 for _ in range(self._n_layers)]
        self.__z = [0 for _ in range(self._n_layers)]
        self.__mp_indices = [0 for _ in range(self._n_layers)] 
       
        for l in range(self._n_layers):
            
            if self.layers[l] == 'C':
                
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
                    
                    print(f'Layer type: {self.layers[l]}')
                    print(f"Input Shape at Layer {l+1}: {self.X.shape}")
                    print(f"Activation Func {act_func}")
                    print(f"Stride at Layer {l+1}: {stride}")
                    print(f"Kernel Shape at Layer {l+1}: {kernel.shape}")
                    print(f"Bias shape at layer {self.bias[l].shape}") 
                    print(f"Output Shape at Layer {l+1}: {self.__a[l].shape}\n")
                else:
                    #print(f"Input Shape at Layer {l+1}: {a.shape}")
                    self.__z[l] = cops.conv2D(self.__a[l - 1], kernel, stride) + b
                    self.__a[l] = act_func(self.__z[l])
                    print(f'Layer type: {self.layers[l]}')
                    print(f"Input Shape at Layer {l+1}: {self.__a[l - 1].shape}")
                    print(f"Activation Func {act_func}")
                    print(f"Stride at Layer {l+1}: {stride}")
                    print(f"Kernel Shape at Layer {l+1}: {kernel.shape}")
                    print(f"Bias shape at layer {self.bias[l].shape}") 
                    print(f"Output Shape at Layer {l+1}: {self.__a[l].shape}\n")
      
            elif self.layers[l] == 'MP':
                
                kernel = self.weights[l] 
                stride = self.stride[l] 
               
                if l == 0:
                    self.__a[l], self.__mp_indices[l] = cops.maxpool2D(self.X, kernel, stride, self.n_output_channels[l], return_idxs = True )
                    self.__mp_indices[l] = self.__mp_indices[l].astype(int)

                    print(f'Layer type: {self.layers[l]}') 
                    print(f"Input Shape at Layer {l+1}: {self.X.shape}")
                    print(f"Activation Func {act_func}")
                    print(f"Stride at Layer {l+1}: {stride}")
                    print(f"Kernel Shape at Layer {l+1}: {kernel}")
                    print(f"Output Shape at Layer {l+1}: {self.__a[l].shape}\n") 
                    
                else:

                   
                    self.__a[l], self.__mp_indices[l] = cops.maxpool2D(self.__a[l - 1], kernel, stride, self.n_output_channels[l], return_idxs = True) 
                    self.__mp_indices[l] = self.__mp_indices[l].astype(int)

                    print(f'Layer type: {self.layers[l]}')                    
                    print(f"Input Shape at Layer {l+1}: {self.__a[l - 1].shape}")
                    print(f"Activation Func {act_func}")
                    print(f"Stride at Layer {l+1}: {stride}")
                    print(f"Kernel Shape at Layer {l+1}: {kernel}")
                    print(f"Output Shape at Layer {l+1}: {self.__a[l].shape}\n") 
        
            elif self.layers[l] == 'AP':
                
                kernel = self.weights[l]
                stride = self.stride[l]
                
                if l == 0:
                    self.__a[l] = cops.avgpool2D(self.X, kernel, stride, self.n_output_channels[l]) 
                    
                    print(f'Layer type: {self.layers[l]}')                    
                    print(f"Input Shape at Layer {l+1}: {self.X.shape}")
                    print(f"Activation Func {act_func}")
                    print(f"Stride at Layer {l+1}: {stride}")
                    print(f"Kernel Shape at Layer {l+1}: {kernel}")
                    print(f"Output Shape at Layer {l+1}: {self.__a[l].shape}\n") 
                else:
                    self.__a[l] = cops.avgpool2D(self.__a[l - 1], kernel, stride, self.n_outptu_channels[l])
                    
                    print(f'Layer type: {self.layers[l]}')                    
                    print(f"Input Shape at Layer {l+1}: {self.__a[l - 1].shape}")
                    print(f"Activation Func {act_func}")
                    print(f"Stride at Layer {l+1}: {stride}")
                    print(f"Kernel Shape at Layer {l+1}: {kernel}")
                    print(f"Output Shape at Layer {l+1}: {self.__a[l].shape}\n")             
                    
            elif self.layers[l] == 'FC': 
                
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
                    
                print(f'Layer type: {self.layers[l]}')  
                print(f"Input Shape at Layer {l+1}: {self.__a[l - 1].shape}")
                print(f"Activation Func {act_func}")
                print(f"Weights shape at layer {self.weights[l].shape}")
                print(f"Bias shape at layer {self.bias[l].shape}") 
                print(f"output shape at layer: {l+1}: {self.__a[l].shape}\n")

        return self.__a[-1]
        
    def _backward(self):

        self.__dz = [np.zeros_like(self.__a[l]) if l in ['FC', 'C'] else 0 for l, _ in enumerate(self.layers)]
        self.__dw = [np.zeros_like(self.__a[l]) if l in ['FC', 'C'] else 0 for l, _ in enumerate(self.layers)]
        self.__db = [np.zeros_like(self.__a[l]) if l in ['FC', 'C'] else 0 for l, _ in enumerate(self.layers)]
     
        batch_size = self.X.shape[0] 
      
        for l, _ in reversed(list(enumerate(self.layers))):
            
            act_func = F.get_func(self.activations[l]) 

            print(f"\nLayer: {l + 1}") 
            print(f"Layer idx: {l}")
            
            if self.layers[l] == 'FC':
                if l == (self._n_layers - 1): # if we're at the last layer
                    self.__dz[l], self.__dw[l], self.__db[l] = fcops.grad_last_layer(self.y_onehot, 
                                                                                     self.__a, 
                                                                                     batch_size, 
                                                                                     l) 
                    print(f"dz: {self.__dz[l].shape}")
                    print(f"dw: {self.__dw[l].shape}")
                    print(f'db: {self.__db[l].shape}')
                elif self.layers[l - 1] in ['MP', 'AP', 'C']:
                    
                    self.__dz[l], self.__dw[l], self.__db[l] = fcops.grad_cblock_prior(self.__a, 
                                                                                       self.weights, 
                                                                                       self.__z, 
                                                                                       batch_size, 
                                                                                       act_func, 
                                                                                       l)
                    
                    print(f"dz: {self.__dz[l].shape}")
                    print(f"dw: {self.__dw[l].shape}")
                    print(f'db: {self.__db[l].shape}')        
                    
                else:
                    
                    self.__dz[l], self.__dw[l], self.__db[l] = fcops.grad_fcblock(self.__a, 
                                                                                       self.weights, 
                                                                                       self.__z, 
                                                                                       batch_size, 
                                                                                       act_func, 
                                                                                       l) 
                    
                    print(f"dz: {self.__dz[l].shape}")
                    print(f"dw: {self.__dw[l].shape}")
                    print(f'db: {self.__db[l].shape}')        
                    
            elif self.layers[l] == 'C':
                
                '''
               
                TODO - implement backprop for conv layers 
                
                '''
                
                print("Grad Conv")
              
                if self.layers[l + 1] == 'FC':
                    '''TODO'''
                    pass
                elif self.layers[l + 1] == 'C':
                    '''TODO (do i need it?)'''
                    
                    self.__dz[l] = cops.grad_conv2D() 
                    
                elif self.layers[l+ 1] == 'MP':
                 
                    self.__dz[l] = cops.grad_conv2D_from_maxpool(self.__dz[l+1], self.__z[l], self.__mp_indices[l + 1], act_func)
                    print(f'dz: {self.__dz[l].shape}') 
                    
                elif self.layers[l + 1] == 'AP':
                    '''TODO ( do i need this or can i integrate with 'MP'?)'''
                
                
               
            elif self.layers[l] == 'MP':
                
                '''
                TODO - implement backprop for maxpool layers
                '''

                print('Grad MaxPool')
                
                if self.layers[l + 1] == 'FC': 
                   
                    self.__dz[l] = cops.grad_maxpool2D_from_fc(self.weights[l + 1], self.__z[l+1], self.__a[l])
                    print(f'dz: {self.__dz[l].shape}')

                elif self.layers[l + 1] == 'C': # if the layer after the maxpool layer was the conv layer.
                    # TODO
                    pass
                    
            elif self.layers[l] == 'AP':
                
                '''
               
                TODO - implement backprop for avgpool layers:
                  
                "You can compute the gradients of a global avg pool w.r.t the inputs - it's simply dividing by the number of elements pooled. 
              
                https://stackoverflow.com/questions/70691539/how-does-error-get-back-propagated-through-pooling-layers               
                https://stats.stackexchange.com/questions/565032/cnn-upsampling-backprop-gradients-across-average-pooling-layer  
               
                DON'T FORGET ABOUT SCENEARIOS WHERE THE KERNEL DOESN'T FIT ONTO THE FEATURE MAP FOR EVERY STRIDE
                                
                '''

                print('Grad AvgPool')

        return
    
    def _init_params(self):
        
        weights = []
        bias = [] 
        conv_out_dims = [] 
        
        for l, l_type in enumerate(self.layers):
            if l_type.upper() == 'C':
                
                print(f"layer {l} is type {l_type}")
               
                _bool = self._check_dilation(l) # True if dilation is set for the lth conv layer, else is False
                
                if not _bool: # Default kernel creation if dilation is not expected
                   
                    weights.append(self._init_kernel(l))
                
                elif _bool: # Dilated kernel if dilation is expected
                    
                    kernel_mask = self._init_kernel(l)
                    weights.append(self._dilate_kernel(kernel_mask, l))
                
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
                
                print(f"layer {l} is type {l_type}") 
             
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
                print(f"layer {l} is type {l_type}") 
                weights.append(self._init_fc(l, conv_out_dims))
                bias.append(np.zeros(shape = (self.layer_size[l], 1))) 
      
        return weights, bias   
    
     
    def _pad(self, X, padding):
        return np.pad(X, pad_width = ((0, 0), (0, 0), (padding[0], padding[0]), (padding[0], padding[0])))

    def _dilate_kernel(self, kernel_mask, l):
        '''
        kernel_mask: shape (Out Channels, In Channels, H, W)
        l: the lth layer
        '''
        
        print(kernel_mask.shape)
        
        if kernel_mask.shape[2] == 1 and kernel_mask.shape[3] == 1:
            return kernel_mask
        
        d_h = self.dilation_rate[l][0] 
        d_w = self.dilation_rate[l][1] 
        
        k_h = (kernel_mask.shape[2] - 1) * d_h + 1
        k_w = (kernel_mask.shape[3] - 1) * d_w + 1
        
        out_kernel = np.zeros((kernel_mask.shape[0], kernel_mask.shape[1], k_h, k_w))
        
        for out_ch in range(kernel_mask.shape[0]):
            for in_ch in range(kernel_mask.shape[1]):
                for row in range(kernel_mask.shape[2]):
                    for col in range(kernel_mask.shape[3]):
                        out_row = row * d_h
                        out_col = col * d_w
                        
                        out_kernel[out_ch, in_ch, out_row, out_col] = kernel_mask[out_ch, in_ch, row, col]
        
        return out_kernel
 
    def _init_kernel(self, l):
        
        # TODO -- fix kernel initializing for max pooling and avg pooling -- the 'kernel' will be kept as a tuple of (H, W), will not be an ndarray of zeros
        # -- fix self.layer_size for max pooling / avg pooling layers and also for initializing the params -- some in layer_size will be 'nan' so need to handle that so i don't get an error for the lth layer. 
         
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
                
                print(conv_out_dims)
                h, w = conv_out_dims[l - 1] 
                conv_out_size = self.n_output_channels[l - 1] * h * w
                print(self.layer_size[l])                 
                weights = np.random.randn(self.layer_size[l], conv_out_size ) * np.sqrt(1/(conv_out_size))            
                
            else:
#                print(self.layer_size[l-1])
#                print(self.layer_size[l])
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
    def y_onehot(self):
        return self._y_onehot

    @y_onehot.setter
    def y_onehot(self, y_onehot):
        if y_onehot is None:
            y_onehot = E.one_hot_encode(self.y) 
        self._y_onehot = y_onehot
            
if __name__ == "__main__":
    
    seed = 1
  
    X = np.random.random_sample(size = (2, 2, 2, 2))
    
    padding = ((1, 1), 1, (0, 0)) 
    
    nn = ConvNet(seed = seed)