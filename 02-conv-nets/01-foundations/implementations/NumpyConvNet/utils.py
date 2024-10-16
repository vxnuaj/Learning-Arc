import numpy
import math

class Encoding:
    
    @staticmethod
    def one_hot_encode(y): 
        y_onehot = numpy.zeros((numpy.max(y) + 1, y.size)) # 3 classes, 3 labels
        y_onehot[y, numpy.arange(y.size)] = 1
        return y_onehot

class Transforms:
    
    @staticmethod
    def dilate_kernel(kernel_mask, dilation_rate):
        '''
        kernel_mask: shape (Out Channels, In Channels, H, W)
        l: the lth layer
        '''
        
        if kernel_mask.shape[2] == 1 and kernel_mask.shape[3] == 1:
            return kernel_mask
        
        d_h = dilation_rate[0] 
        d_w = dilation_rate[1] 
        
        k_h = (kernel_mask.shape[2] - 1) * d_h + 1
        k_w = (kernel_mask.shape[3] - 1) * d_w + 1
    
        out_kernel = numpy.zeros((kernel_mask.shape[0], kernel_mask.shape[1], k_h, k_w))
        
        for out_ch in range(kernel_mask.shape[0]):
            for in_ch in range(kernel_mask.shape[1]):
                for row in range(kernel_mask.shape[2]):
                    for col in range(kernel_mask.shape[3]):
                        out_row = row * d_h
                        out_col = col * d_w
                        
                        out_kernel[out_ch, in_ch, out_row, out_col] = kernel_mask[out_ch, in_ch, row, col]
        
        return out_kernel 

    @staticmethod
    def pad(X, padding):
        return numpy.pad(X, pad_width = ((0, 0), (0, 0), (padding[0], padding[0]), (padding[0], padding[0])))

         
def validate_inputs(
       
        X: numpy.ndarray,
        y: numpy.ndarray, 
        layers: list[str], 
        output_channels: list[int], 
        layer_size: list[tuple[int, int] | int], 
        activations: list[str] | str,
        y_onehot:numpy.ndarray,
        padding: list[tuple[int, int] | int], 
        stride: list[tuple[int, int] | int], 
        dilation_rate: list[tuple[int, int] | int],
        param_init: list[str],
        n_layers: int,
        n_conv_layers: int,
        n_conv_block_layers: int,
        n_fc_layers: int,
        alpha:int|float,
        epochs: int
        
    ):
   
        assert isinstance(X, numpy.ndarray), 'X must be a 4-dimensional numpy array.'
        assert len(X.shape) == 4, 'X must be shape (B, C, H, W)'
        
        assert isinstance(y, numpy.ndarray), 'y must be an 1-dimensional numpy array'
        assert len(y.shape) == 1 or len(y.shape) == 2, 'y must be shape (B, 1) or simply (B, )'
    
        assert isinstance(layers, list), "layers must be a list"
        for i, layer in enumerate(layers):
            assert isinstance(layer, str), f"Expected string in layers at index {i}, got {type(layer)}"
            assert layer in ['C', 'FC', 'MP', 'AP'], f'Layer at index {i} is not defined properly. Got {layer}, can only be C, FC, MP, or AP'

        assert isinstance(output_channels, list), "output_channels must be a list"
        assert len(output_channels) == n_conv_layers, "the amount of defined output channels in output_channels must be the same amount as the amount of your convolutional layers"
        for channel in output_channels:
            assert isinstance(channel, (tuple, int)), f"Expected tuple or int in output_channels, got {type(channel)}"
            
        assert isinstance(layer_size, list), "layer_size must be a list"
        assert len(layer_size) == n_layers, "the amount of defined layer_size must be the same amount as the total amount of layers"
        for param_size in layer_size:
            assert isinstance(param_size, (tuple, int)) or (isinstance(param_size, float) and math.isnan(param_size)), "param_size must be a tuple, int, or NaN"
            if isinstance(param_size, tuple):
                assert len(param_size) == 2 and all(isinstance(i, int) for i in param_size), "Each tuple in layer_size must contain two integers"
             
        assert isinstance(activations, (str, list)), 'activations must be type list (for multiple) or str (for single for the entire neural net)'
        if isinstance(activations, list):
            n =  n_layers - len(activations)
            l = [0 for _ in layers if _ in ['C', 'FC']]
            assert len(activations) == len(l),\
            f"unless you're defining a single activation function for the entire neural net, you must define activations for all layers. you either have too many or too little defined"
            for func in activations:
                assert func in ['relu', 'leakyrelu', 'parameterizedrelu', 'tanh', 'sigmoid', 'softmax'], ValueError("function is invalid. must be one of the following: 'relu', 'leakyrelu', 'parameterizedrelu', 'tanh', 'sigmoid', 'softmax'")
        elif activations.lower() not in ['relu', 'leakyrelu', 'parameterizedrelu', 'tanh', 'sigmoid', 'softmax']:
            raise ValueError("the defined activation function must be one of the following: 'relu', 'leakyrelu', 'parameterizedrelu', 'tanh', 'sigmoid', 'softmax' ")
             
        assert isinstance(y_onehot, (numpy.ndarray, type(None))), 'y_onehot must be a 2-dimensional numpy array'
        if isinstance(y_onehot, numpy.ndarray):
            assert len(y_onehot.shape) == 2 , 'y must be shape (Classes, B)'
                    
        if padding != 0:
            assert isinstance(padding, list), "padding must be a list"
            assert len(padding) == n_conv_block_layers, 'the amount of defined padding sizes must be the same amount as the amount of layers in the convolution block.'
            for pad in padding:
                assert isinstance(pad, (tuple, int)), f"Expected tuple or int in padding, got {type(pad)}"
                if isinstance(pad, tuple):
                    assert len(pad) == 2 and all(isinstance(i, int) for i in pad), "Each tuple in padding must contain two integers"

        if stride != 1:
            assert isinstance(stride, list), "stride must be a list"
            assert len(stride) == n_conv_block_layers, 'The amount of defined stride lengths must match the amount of layers in the coonvolution block'
            for s in stride:
                assert isinstance(s, (tuple, int)), f"Expected tuple of length 2 or int in stride, got {type(s)}"
                if isinstance(s, tuple):
                    assert len(s) == 2, f"Expected tuple of length 2 in stride, got tuple of length {len(s)}"
                    assert all(isinstance(i, int) for i in s), "Each element in the stride tuple must be an integer."

        if dilation_rate != 1:
                
            assert isinstance(dilation_rate, list), "dilation_rate must be a list"
            assert len(dilation_rate) == n_conv_layers, 'the amount of defined dilation_rate must be the same amount as the amount of convolutional layers.'
            for i, dilation in enumerate(dilation_rate):
                assert isinstance(dilation, (tuple, int)), f"Expected tuple or int in dilation_rate, got {type(dilation)}"
                if isinstance(dilation, tuple):
                    assert len(dilation) == 2 and all(isinstance(i, int) for i in dilation), "Each tuple in dilation_rate must contain two integers"
                    for j, val in enumerate(dilation):
                        assert isinstance(val, int), f'Dilation rates must be type int, got {type(val)} instead.' 
                        assert val >= 1, f"Dilation rate must be at least 1 if defined, instead got {val} at tuple index {i}, element index {j}"
                
                if isinstance(dilation, (int, float)):
                    assert isinstance(dilation, int), f'Dilation rate must be type int, got {type(dilation)} at index {i}'
                    assert dilation >= 1, f"Dilation rate must be at least 1 if defined, instead got {dilation} at index {i}"
       
        assert isinstance(param_init, str), 'param_init must be type str, of "glorot" or "kaiming".'
        assert param_init in ['glorot', 'kaiming'], 'param_init must be "glorot" or "kaiming".'
        
        assert isinstance(alpha, (float, int)), 'alpha must be type int or float'
        
        assert isinstance(epochs, int), 'epochs must be type int'
        assert epochs > 0, 'you cannot have negative epochs. might as well not train your model!'