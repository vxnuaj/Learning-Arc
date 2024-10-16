import numpy as np

class FCOps:
    
    @staticmethod
    def grad_last_layer(y_onehot, a, batch_size, l):
        dz = a[l] - y_onehot
        dw = np.dot(dz, a[l - 1].T) / batch_size
        db = np.mean(dz, axis = 1, keepdims = True)
        return dz, dw, db

    def grad_cblock_prior(a, w, z, batch_size, act_func, l):
        
        a = a[l-1].reshape(batch_size, -1) 
       
        dz = np.dot(w[l + 1].T, z[l + 1]) * act_func(z[l], deriv = True)
        dw = np.dot(dz, a) / batch_size
        db = np.mean(dz, axis = 1, keepdims = True)
        return dz, dw, db

    def grad_fcblock(a, w, z, batch_size, act_func, l):
        
        dz = np.dot(w[l+1].T, z[l+1]) * act_func(z[l], deriv = True)
        dw = np.dot(dz, a[l-1].T) / batch_size
        db = np.mean(dz, axis = 1, keepdims=True)
        return dz, dw, db

class ConvOps:
    @staticmethod
    def conv2D(X, kernel, stride):
        
        '''
        X: input for the lth layer, shape: (B, C, H, W)
        Kernel: kernel weights for the lth layer  (Channels out, Channels in, H, W)
        Y (output): output of Kernel convolved over X, shape: (B, C, H, W) 
        
        ''' 
      
        out_h = int(((X.shape[2] - kernel.shape[2]) / stride[0]) + 1)
        out_w = int(((X.shape[3] - kernel.shape[3]) / stride[1]) + 1)
        
        Y = np.zeros(shape = (X.shape[0], kernel.shape[0], out_h, out_w)) # output shape: (B, C, H, W) 

        for sample in range(X.shape[0]): # for each sample
            for out_ch in range(Y.shape[1]): # for each set of kernels or for each output channel
                for m in range(Y.shape[2]): # for the rows of the kernel size
                    for n in range(Y.shape[3]): # for the height of the kernel size
                        for in_ch in range(X.shape[1]): # for the amount of input channels
                            m_idx = m * stride[0]
                            n_idx = n * stride[1]
                            
                            patch = X[

                                sample, 
                                in_ch, 
                                m_idx: (m_idx + kernel.shape[2]), 
                                n_idx: (n_idx + kernel.shape[3])
                                
                                ]
                            
                            
                            if patch.size != kernel[out_ch, in_ch].size:
                                break 
                            
                            conv_ch = patch * kernel[out_ch, in_ch]
                                
                            Y[sample, out_ch, m, n] = np.sum(conv_ch)
                            
                            
        return Y
    
    @staticmethod
    def transposed_conv2D(Z: np.ndarray, kernel: np.ndarray, stride, output_ch, out_shape = None):
        
        batch_size = Z.shape[0]
        
      
        kernel = np.flip(kernel, axis=(2, 3)) 
        
        k_h = kernel.shape[2] 
        k_w = kernel.shape[3] 
        s_h = stride[0]  
        s_w = stride[1]  

        in_ch = Z.shape[1]
        i_h = Z.shape[2]  
        i_w = Z.shape[3] 
      
        out_h = s_h * (i_h - 1) + k_h  
        out_w = s_w * (i_w - 1) + k_w  
       
        Y = np.zeros(shape=(batch_size, output_ch, out_h, out_w))  
        
        for sample in range(batch_size): 
            for o_ch in range(output_ch):  
                for in_ch_idx in range(in_ch):  
                    for m in range(i_h):  
                        for n in range(i_w): 
                            
                            y_start_h = m * s_h
                            y_end_h = y_start_h + k_h
                            y_start_w = n * s_w
                            y_end_w = y_start_w + k_w
                            
                            Y[sample, o_ch, y_start_h:y_end_h, y_start_w:y_end_w] += (
                                kernel[in_ch_idx, o_ch] * Z[sample, in_ch_idx, m, n]
                            )
      
        if out_shape is not None and Y.shape[2:4] != out_shape[2:4]: 
            if Y.shape[2] != out_shape[2]:
                pad = np.abs(Y.shape[2] - out_shape[2])
                Y = np.pad(Y, pad_width = ((0, 0), (0, 0), (0, pad), (0, 0)) )
            elif Y.shape[3] != out_shape[3]:
                pad = np.abs(Y.shape[3] - out_shape[3])
                Y = np.pad(Y, pad_width = ((0, 0), (0, 0), (0, 0), (0, pad)))
            elif Y.shape[2] != out_shape[2] and Y.shape[3] != out_shape[3]:
                pad_b = np.abs(Y.shape[2] - out_shape[2])
                pad_r = np.abs(Y.shape[3] - out_shape[3])
                Y = np.pad(Y, pad_width = ((0, 0), (0, 0), (0, pad_b), (0, pad_r)))
                
        return Y 

    @staticmethod
    def maxpool2D(Z, kernel, stride, output_ch, return_idxs = False):
          
        k_h = kernel[0]
        k_w = kernel[1] 
        s_h = stride[0]
        s_w = stride[1]
        i_h = Z.shape[2]
        i_w = Z.shape[3]
         
        out_h = int(((i_h - k_h) / s_h) + 1)
        out_w = int(((i_w - k_w) / s_w) + 1)
        
        Y = np.zeros(shape = (Z.shape[0], output_ch, out_h, out_w))
        
        if return_idxs:
            indices = np.zeros_like(Y)
        
        for sample in range(Z.shape[0]):
            for out_ch in range(Y.shape[1]):
                for m in range(Y.shape[2]) :
                    for n in range(Y.shape[3]):
                        for in_ch in range(Z.shape[1]):
                           
                            n_idx = n * s_w
                            m_idx = m * s_h
                            
                            patch = Z[
                               
                                sample,
                                in_ch,
                                m_idx: (m_idx + k_h),
                                n_idx: (n_idx + k_w)  
                                
                                ]
                            
                            if patch.size != (k_h * k_w):
                                break
                            
                            Y[sample, out_ch, m, n] = np.max(patch)
                            
                            if return_idxs:
                                
                                max_idx = np.argmax(patch) 
                                
                                max_idx_h = max_idx // k_w
                                max_idx_w = max_idx % k_w 
                               
                                orig_idx_h = m_idx + max_idx_h 
                                orig_idx_w = n_idx + max_idx_w
                                
                                indices[sample, out_ch, m, n] = orig_idx_h * i_w + orig_idx_w
       
       
        if return_idxs:
            return Y, indices
        
        return Y
    
    @staticmethod
    def avgpool2D(Z, kernel, stride, output_ch):
          
        k_h = kernel[0] 
        k_w = kernel[1]
        s_h = stride[0]
        s_w = stride[1]
        i_h = Z.shape[2]
        i_w = Z.shape[3]
         
           
        out_h = int(((i_h - k_h) / stride[0]) + 1)
        out_w = int(((i_w - k_w) / stride[1]) + 1)
        
        Y = np.zeros(shape = (Z.shape[0], output_ch, out_h, out_w))
       
        for sample in range(Z.shape[0]):
            for out_ch in range(Y.shape[1]):
                for m in range(Y.shape[2]) :
                    for n in range(Y.shape[3]):
                        for in_ch in range(Z.shape[1]):
                            
                            n_idx = n * s_h
                            m_idx = m * s_w
                            
                            patch = Z[
                               
                                sample,
                                in_ch,
                                m_idx: (m_idx + k_h),
                                n_idx: (n_idx + k_w)  
                                
                                ]
                            
                            if patch.size != (k_h * k_w):
                                break
                            
                            Y[sample, out_ch, m, n] = np.mean(patch)
                            
                            
       
        return Y
    
    @staticmethod
    def grad_conv2D_from_maxpool(dz, z, X, indices, act_func, stride, kernel):
       
        '''
        dz: gradient being propagated back from layer l + 1
        z:  weighted sum at layer l
        X: input to layer l
        indices: flattened indices of maxpooled values during forward prop (B, C, H, W | H, W, C correspond to same dims as maxpooling layer
        act_func: activation function at layer l 
        '''
      
        in_ch = dz.shape[1]
        batch_size = dz.shape[0]
        dz_m = dz.shape[2]
        dz_n = dz.shape[3]
        
        dz_out = np.zeros(z.shape) # making the correpsonding deriviative the same size as it's forwardprop counterpart
        batch_size = z.shape[0] 
        
        for sample in range(batch_size):
            for in_ch in range(in_ch):
                for m in range(dz_m):
                    for n in range(dz_n):
                        
                        flat_idx = indices[sample, in_ch, m, n]
                        orig_idx_h, orig_idx_w = np.unravel_index(flat_idx, (z.shape[2], z.shape[3]))
                        
                        dz_out[sample, in_ch, orig_idx_h, orig_idx_w] += dz[sample, in_ch, m, n]
       
        
        dz_out *= act_func(z, deriv = True) 
        dw = ConvOps.__dw_grad_conv2D_util(dz_out, X, stride, batch_size = batch_size, kernel_shape = kernel.shape)
        db = (np.sum(dz_out, axis = (0, 2, 3)) / batch_size).reshape(kernel.shape[0], 1, 1)
        
        return dz_out, dw, db
  
    @staticmethod
    def grad_conv2D_from_avgpool(dz, z, X, kernel, stride, act_func):
        
        '''
        dz: gradient being propagated back from layer l + 1
        
        z: weighted sum at layer l (used to match dims for mask)
        
        X: input to layer l
        
        act_func: activation function at layer l 
        
        '''  

        in_ch = dz.shape[1] 
        batch_size = dz.shape[0]
        dz_m = dz.shape[2] 
        dz_n = dz.shape[3]
        dz_out = np.zeros(z.shape)      
       
        s_h = stride[0]
        s_w = stride[1]
         
        
        k_h = kernel.shape[2]
        k_w = kernel.shape[3]

        batch_size = z.shape[0]
        patch_size = (k_h * k_w)
        
        for sample in range(batch_size):
            for in_ch in range(in_ch):
                for m in range(dz_m):
                    for n in range(dz_n):
                        
                        dz_out[sample, in_ch, m * s_h: (m * s_h) + k_h, n * s_w: (m * s_w) + k_w] += dz[sample, in_ch, m, n] / patch_size
  
        dz_out *= act_func(z, deriv = True)
        dw = ConvOps.__dw_grad_conv2D_util(dz_out, X, stride, batch_size, kernel.shape)
        db = (np.sum(dz_out, axis = (0, 2, 3)) / batch_size).reshape(-1, 1, 1)
       
        return dz_out, dw, db
    

  
   
    @staticmethod
    def grad_convblock_from_fc(w, z_1 = None, z_2 = None, a_0 = None, a_1 = None, kernel = None, stride = None, act_func = None, layer = 'C'):
     
        '''
        
        w: Weights at the l + 1 layer
        
        z: Output at the l + 1 layer
        
        z_1: Weighted sum of the current layer
        
        z_2: Weighted sum at the l + 1 layer
        
        a_0: Input to the current layer
        
        a_1: Activation at the current layer
      
        kernel: Kernel at the current layer 
       
        stride: stride for the current layer (only if in convblock) 
      
        layer: the current layer 
       
        act_func: the activation function used at the current layer (if applicable) 
        
        '''
     
        if layer == 'C':
    
            batch_size = z_1.shape[0]
            
            dz = np.dot(w.T, z_2)
            dz = dz.reshape(*z_1.shape) * act_func(z_1, deriv = True) 
            
            dw = ConvOps.__dw_grad_conv2D_util(dz, a_0, stride, batch_size, kernel.shape )
            db = (np.sum(dz, axis = (0, 2, 3)) / batch_size).reshape(-1, 1, 1)
           
            return dz, dw, db 
            
        elif layer in ['MP', 'AP']:
            dz = np.dot(w.T, z_2)
            dz = dz.reshape(*a_1.shape) 
            return dz
   
    @staticmethod
    def grad_maxpool2D(dZ, kernel, stride, output_ch, out_shape):
        
        '''
       
        Gradient for max-pooling layer, if located right before a convolutional layer
        
        dZ: gradient being propagated back from conv layer
        
        kernel: the kernel from layer l+1 
        
        stride: stride of the current layer
        
        output_ch: # of channels in the output of the current layer
        
        ''' 
       
        dz = ConvOps.transposed_conv2D(dZ, kernel, stride, output_ch, out_shape) 
      
        return dz 
    
    @staticmethod
    def grad_avgpool2D(dz, kernel, stride, output_ch, out_shape):
      
        '''
        
        Gradient for the average-poooling layer, if located right before a convolutional layer 
       
        
        ''' 
        
        dz = ConvOps.transposed_conv2D(dz, kernel, stride, output_ch, out_shape) 
        
         
        return dz
    
    @staticmethod
    def grad_conv2D(dZ:np.ndarray, z:np.ndarray, X:np.ndarray, kernel_1:np.ndarray, kernel_2:np.ndarray, stride:tuple, output_ch:int, act_func, out_shape):
     
        '''
        dZ: gradient being backpropagated
        
        z: the output of the current layer, in the forward pass
        
        X: the input to the current layer
        
        kernel_1: kernel from the current layer
        
        kernel_2: kernel from layer l+1
        
        kernel: The kenrel of the current layer.
        
        stride: stride of the current layer
        
        output_ch: # of channels in the output of the current layer
        
        act_func: activation function of the current layer.
        '''
        
        batch_size = z.shape[0]
        
        dz = ConvOps.transposed_conv2D(dZ, kernel_2, stride, output_ch, out_shape) * act_func(z, deriv = True)
        dw = ConvOps.__dw_grad_conv2D_util(dz, X, stride, batch_size, kernel_1.shape) 
        db = (np.sum(dz, axis = (0, 2, 3)) / batch_size).reshape(-1, 1, 1)

        return dz, dw, db
   
    def __dw_grad_conv2D_util(dZ, X, stride, batch_size, kernel_shape):
        
        '''
        dZ: the gradient being backpropagated
        X: the input to the current layer
        stride: stride to the current layer. 
        kernel_shape: kernel.shape (np.ndarray.shape)
        '''
        
        dw_out = np.zeros(kernel_shape)
    
        k_h = kernel_shape[2]
        k_w = kernel_shape[3]
    
        dZ_m = dZ.shape[2] 
        dZ_n = dZ.shape[3]  
        
        s_h = stride[0]
        s_w = stride[1]
        
        for sample in range(batch_size): 
            for out_ch in range(dZ.shape[1]):  
                for m in range(dZ_m): 
                    for n in range(dZ_n): 
                        patch = X[
                            sample,
                            :,
                            m * s_h: m * s_h + k_h, 
                            n * s_w: n * s_w + k_w   
                        ]
                         
                        if patch.shape != (k_h * k_w):
                            break 
                        
                        dw_out[out_ch, :, :, :] += patch * dZ[sample, out_ch, m, n]
        
        return dw_out / batch_size
