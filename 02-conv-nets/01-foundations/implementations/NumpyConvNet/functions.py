import numpy as np

class Functions:
 
    '''
   
    Activation Functions & Loss Functions 
    
    '''
  
    eps = 1e-10
   
    def __init__(self):
        pass
   
    @staticmethod
    def get_func(func):
        if func == 'relu':
            return Functions.ReLU
        elif func == 'leakyrelu':
            return Functions.LeakyReLU
        elif func == 'parameterizedrelu':
            return Functions.ParameterizedReLU
        elif func == 'tanh':
            return Functions.Tanh
        elif func == 'sigmoid':
            return Functions.Sigmoid
        elif func == 'softmax':
            return Functions.Softmax
   
    @staticmethod
    def ReLU(z, deriv = False):
        if deriv == True:
            return np.where(z > 0, 1, 0)
        return np.where(z > 0, z, 0)

    @staticmethod 
    def LeakyReLU(z, deriv = False):
        if deriv == True:
            return np.where(z > 0, 1, .01)
        return np.where(z > 0, z, z * .01)

    @staticmethod        
    def ParameterizedReLU(z, alpha, deriv = False):
        if deriv == True:
            return np.where(z > 0, 1, alpha)
        return np.where(z > 0, z, z * alpha)
 
    @staticmethod
    def Tanh(z, deriv = False):
        a = (np.exp(z + Functions.eps) - np.exp(-z + Functions.eps )) / (np.exp(z + Functions.eps) + np.exp(-z + Functions.eps))
        if deriv == True:
            return 1 - np.square(a)
        return a 
   
    @staticmethod 
    def Sigmoid(z, deriv = False):
        a = 1 / 1 + np.exp(-z + Functions.eps)
        if deriv == True:
            return a * (1 - a)
        return a
      
    @staticmethod 
    def Softmax(z):
        return np.exp(z + Functions.eps) / np.sum(np.exp(z + Functions.eps), axis = 0)
    

    @staticmethod
    def CrossEntropyLoss(y, prob ):
        return - np.sum(y * np.log(prob + Functions.eps)) / y.size
    
    @staticmethod
    def Accuracy(y, pred):
        return np.sum(y == pred) / y.size * 100