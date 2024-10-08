import numpy as np

class Conv:
    def __init__(self, seed = None):
        if seed is not None:
            self.seed = seed
        
    def Conv1D(self, X, kernel_size: int):
        self.X = X
        self.kernel_size = kernel_size
        self.kernel = np.random.random_sample(size = (kernel_size))
        X_size = self.X.size
        output_width = X_size - kernel_size + 1
        Y = np.zeros(shape = (output_width))
        for i in range(Y.size):
            Y[i] = np.sum(self.X[i:i + kernel_size] * self.kernel)
        return Y
  
    def Conv1D_pad(self, X, kernel_size, padding_size):
        self.padding_size = padding_size
        self.X = np.pad(X, pad_width = padding_size)
        self.kernel_size = kernel_size
        self.kernel = np.random.random_sample(size = (kernel_size))
        X_size = self.X.size
        output_width = X_size - kernel_size + 1
        Y = np.zeros(shape = (output_width))
        for i in range(Y.size):
            Y[i] = np.sum(self.X[i:i + kernel_size] * self.kernel)
        return Y
     
    def Conv1D_pad_stride(self, X, kernel_size, padding_size, stride):
        self.padding_size = padding_size
        self.stride = stride
        self.X = np.pad(X, pad_width = padding_size)
        self.kernel_size = kernel_size
        self.kernel = np.random.random_sample(size = (kernel_size))
       
        X_size = self.X.size
        
        output_size = int(((X_size + (2 * padding_size)- kernel_size) / (stride)) + 1)
        Y = np.zeros(shape = (output_size))
      
        for i in range(Y.size):
            Y[i] = np.sum(self.X[i * self.stride:i * self.stride +kernel_size] * self.kernel)
        return Y
  
    def Conv1D_pad_stride_dilation(self, X, kernel_size, padding_size, stride, dilation_rate):
        self.padding_size = padding_size
        self.stride = stride
        self.dilation_rate = dilation_rate
        self.X = np.pad(X, pad_width=padding_size)

        kernel_mask = np.random.random_sample(size=(kernel_size))
        self.kernel = self._dilate_1D(kernel_mask, dilation_rate=self.dilation_rate)
        
        output_size = int(((self.X.size - self.kernel.size) / self.stride) + 1)
        Y = np.zeros(shape=(output_size))

        for i in range(Y.size):
            current_slice = self.X[i * self.stride : (i * self.stride + self.kernel.size)]

            Y[i] = np.sum(current_slice * self.kernel) 

        return Y  
   
    def _dilate_1D(self, kernel, dilation_rate = 2):
        dilation_rate -= 1
        i = 0
        
        if len(kernel) == 1:
            return kernel
        
        while i < len(kernel):
            if kernel[i] != 0:
                kernel = np.concatenate((kernel[:i+1], [0 for _ in range(dilation_rate)], kernel[i+1:])) # works instead of X[i+1:], as we want to exclude the ith element for the end. we already including that in the former part of the X array.
                i += dilation_rate
            i += 1
            if i == (len(kernel) - 1):
                return kernel
    
    @property
    def seed(self):
        return self._seed
    
    @seed.setter
    def seed(self, seed):
        np.random.seed(seed = seed)
        self._seed = seed
    
    @property
    def kernel_size(self):
        return self._kernel_size     
   
    @kernel_size.setter
    def kernel_size(self, kernel_size):
        assert self.X.size >= kernel_size, ValueError('Kernel cannot be greater than input_vector!')
        assert isinstance(kernel_size, int), ValueError('kernel_size must be int for 1D Conv')
        self._kernel_size = kernel_size
       
if __name__ == "__main__":
    
    seed = 1
    ops = Conv(seed = seed)
   
    x = np.array([2, 3, 2, 1, 2])
    print(f"Seed: {seed}") 
    print(f"Input: {x}")
    print()
    Y = ops.Conv1D(X = x, kernel_size=2)
    print(f"ops.Conv1D(X = x, kernel_size=2)\n{Y}")
    print()
    Y = ops.Conv1D_pad(X = x, kernel_size = 2, padding_size=1)
    print(f"ops.Conv1D_pad(X = x, kernel_size = 2, padding_size=1):\n{Y}")
    print()
    Y = ops.Conv1D_pad_stride(X = x, kernel_size = 2, padding_size=0, stride = 2)
    print(f"ops.Conv1D_pad_stride(X = x, kernel_size = 2, padding_size=0, stride = 2):\n{Y}")
    print()
    kernel = np.array([1, 2, 3, 4])
    dilated_kernel =  ops._dilate_1D(kernel = kernel, dilation_rate = 2)
    print(f"ops._dilate_1D(X = x, dilation_rate=2):\n{dilated_kernel}\n")
    
    x = np.array([2, 3, 4, 5, 1, 2, 3, 4, 1, 2])
    print(f"ops.Conv1D_pad_stride_dilation(x, kernel_size=2, padding_size = 2, stride = 2, dilation_rate = 2):")
    Y = ops.Conv1D_pad_stride_dilation(x, kernel_size=2, padding_size = 2, stride = 1, dilation_rate = 2) 
    print(f"{Y}\n")
    