import numpy as np

class FCOps:
    
    @staticmethod
    def grad_last_layer(y_onehot, a, batch_size, l):
        print(len(a))
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
                            
                            
                            #print(f"Patch: {patch}\n") 
                      
                            
                            if patch.size != kernel[out_ch, in_ch].size:
                                break 
                            
                            conv_ch = patch * kernel[out_ch, in_ch]
                                
                            Y[sample, out_ch, m, n] = np.sum(conv_ch)
                            
                            
        return Y
    
    @staticmethod
    def transposed_conv2D(Z:np.ndarray, kernel:np.ndarray, stride, output_ch):
       
        batch_size = Z.shape[0]
        
        kernel = np.flip(kernel, axis = (2, 3)) 
       
        ## ffs this shit took too long. love it though.
        
        k_h = kernel.shape[2] # kernel height
        k_w = kernel.shape[3] # kernel width
        s_h = stride[0] # stride height
        s_w = stride[1] # stride width

        in_ch = Z.shape[1]
        i_h = Z.shape[2] # input height (rows)
        i_w = Z.shape[3] # input width (cols)
      
        out_h = s_h * (i_h - 1) + k_h  # output height
        out_w = s_w * (i_w - 1) + k_w  # output width
        
        Y = np.zeros(shape = (Z.shape[0], output_ch, out_h, out_w))  # output mask
       
        for sample in range(batch_size):  # for each sample
            for out_ch in range(output_ch): # for each output channel (channels in Y)
                for in_ch in range(in_ch): # for each channel in the input
                    for m in range(i_h): # iterate over mth elements in the input, Z
                        for n in range(i_w): # iterate over nth elements in the input, Z

                            y_start_h = m * s_h
                            y_end_h = y_start_h + k_h
                            y_start_w = n * s_w
                            y_end_w = y_start_w + k_w
                            
                            Y[sample, out_ch, y_start_h: y_end_h, y_start_w: y_end_w] += kernel[in_ch, out_ch] * Z[sample, in_ch, m, n]
       
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
       
#        print(f"Z:\n\n{Z}")
       
        if return_idxs:
            return Y, indices
        
        return Y
    
    @staticmethod
    def avgpool2D(Z, kernel, stride, output_ch, return_idxs = False):
          
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
                            
                            
       
#        print(f"Z:\n\n{Z}")
        
        return Y
    
    '''
    @staticmethod
    def grad_maxpool2D(dY:np.ndarray, z:np.ndarray, indices, kernel):
       
        dZ = np.zeros(z.shape) # where Z_shape is the size of the original input to the max pooling layer
        k_h = kernel[0] # get height of max pooling kernel
        k_w = kernel[1] # get width of max pooling kernel
      
        for sample in range(dY.shape[0]):
            for out_ch in range(dY.shape[1]):
                for m in range(dY.shape[2]):
                    for n in range(dY.shape[3]):
                       
                        flat_idx = indices[sample, out_ch, m, n]
                        orig_idx_h, orig_idx_w = np.unravel_index(flat_idx, (z.shape[2], z.shape[3]))
                        dZ[sample, out_ch, orig_idx_h, orig_idx_w] += dY[sample, out_ch, m, n]
        return dZ '''
    
  
    @staticmethod
    def grad_conv2D_from_maxpool(dz, z, indices, act_func):
        
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
       
        return dz_out
   
    @staticmethod
    def grad_maxpool2D_from_fc(w, z, a):
       
        dz = np.dot(w.T, z)
        dz = dz.reshape(10, 2, a.shape[2], a.shape[3])
        
        return dz
    
    @staticmethod
    def grad_avgpool2D(dY, Z, kernel, stride):
        
        dZ = np.zeros(Z.shape)
        k_h = kernel[0]
        k_w = kernel[1]
        
        for sample in range(dY.shape[0]):
            for out_ch in range(dY.shape[1]):
                for m in range(dY.shape[2]):
                    for n in range(dY.shape[3]):
                        for in_ch in range(Z.shape[1]):
                            
                            m_idx = m * stride[0]
                            n_idx = n * stride[1]
                            
                            patch = dZ[
                                      
                                      sample,
                                      in_ch,
                                      m_idx: (m_idx + k_h),
                                      n_idx: (n_idx + k_w) 
                                       
                                       ]
                            
                            if patch.size != (k_h * k_w):
                                break
                            
                            patch += dY[sample, out_ch, m, n] / (k_h * k_w)
        
         
        return dZ        
   
   
    @staticmethod
    def grad_conv2D_from_mp(dZ:np.ndarray, z:np.ndarray, a:np.ndarray, stride, act_func):
       
        '''
        dZ: gradient being propagated backward
        z_prior: logit from the current layer
        a: input to the current layer
        ''' 
      
        batch_size = z.shape[0] 
       
        dz = dZ * act_func(z, deriv = True) # correct 
        dw = ConvOps.__dw_grad_util(dz, a, stride, batch_size)

        
        return dz
    
    @staticmethod
    def grad_conv2D(dZ:np.ndarray, z:np.ndarray, X:np.ndarray, kernel:np.ndarray, stride:tuple, output_ch:int, act_func):
     
        '''
        dZ: gradient being backpropagated
        z: the output of the current layer, in the forward pass
        a: the input to the current layer
        kernel: The kenrel of the current layer.
        stride: stride of the current layer
        output_ch: # of channels in the output of the current layer
        act_func: activation function of the current layer.
        
        '''
      
        batch_size = z.shape[0]
        
        dz = ConvOps.transposed_conv2D(dZ, kernel, stride, output_ch) * act_func(z, deriv = True)
        dw = ConvOps.__dw_grad_util(dz, X, stride, batch_size = batch_size) 

        return dz
    
    def __dw_grad_conv2D_util(dZ, X, stride, batch_size):
    
        '''
        dZ: the gradient being backpropagated
        X: the input to the current layer
        stride: stride to the current layer. 
        
        '''
     
        in_ch = dZ.shape[1]
        dZ_h = dZ.shape[2]
        dZ_w = dZ.shape[3]
        out_ch = X.shape[1]
        X_h = X.shape[2]
        X_w = X.shape[3]
        s_h = stride[0]
        s_w = stride[1]
        
        out_h = int(((dZ_h - X_h) / s_h) + 1)
        out_w = int(((dZ_w - X_w) / s_w) + 1) 
       
        Y = np.zeros(shape = (batch_size, out_ch, out_h, out_w))
        
        for sample in range(batch_size):
            for out_ch in range(out_ch):
                for m in range(out_h):
                    for n in range(out_w):
                        for in_ch in range(in_ch):
                            
                            m_idx = m * s_h
                            n_idx = n * s_w
                            
                            patch = dZ[
                                     
                                        sample,
                                        in_ch,
                                        m_idx: (m_idx + X_h),
                                        n_idx: (n_idx + X_w)  
                                       
                                       ] 
                            
                            if patch.size != X[out_ch, in_ch].size:
                                break
                            
                            conv_ch = patch * X[out_ch, in_ch]
                            
                            Y[sample, out_ch, m, n] = np.sum(conv_ch)
                            
        return Y
                            
                            
                            
                            
                            
                            
                            
                            
                             