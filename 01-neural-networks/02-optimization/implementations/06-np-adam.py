
import numpy as np
import jax as jx
from jax import nn as jnn

from nue.preprocessing import x_y_split, csv_to_numpy, train_test_split, one_hot

import random
import pickle as pkl
import time as t
from termcolor import colored

class NN:
   
    eps = 1e-10 
    
    def __init__(self, train_verbose = False, test_verbose = False, seed = None):
        self.train_verbose = train_verbose
        self.test_verbose = test_verbose
        self.__key = None
        self.seed = seed
    
    def train(self, X_train, Y_train, nn_capacity:dict, alpha = .01, beta_1 = .99, beta_2 = .99, epochs = 250, Y_onehot = None, load_params = False, return_params = True, save_params = False):
        self.X_train = X_train
        self.Y_train = Y_train
        self.nn_capacity = nn_capacity
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epochs = epochs
        self.Y_onehot = Y_onehot
        self.load_params = load_params
        self.return_params = return_params 
        self.save_params = save_params

        self._init_internals() 
        self._init_params()
        self._gradient_descent() 
        self._save_params()      
   
        if self.return_params:
            return self.weights, self.bias 
   
    def test(self, X_test, Y_test, Y_test_onehot = None, load_params = False):
        self.X_test = X_test
        self.Y_test = Y_test
        self.Y_test_onehot = Y_test_onehot
        self.load_params = load_params 
      
        self._get_params()  
        self._inference() 
        
        print(f"Test Accuracy: {self._test_accuracy()}")
        print(f"Test Cross Entropy: {self._test_cross_entropy()}")
        
        return np.argmax(self.activations[-1], axis = 0) 
    
    def _init_params(self):
        
        if self.train_verbose:
            print(colored('Initializing Parameters\n', 'green', attrs = ['bold']))
            
        if self.load_params:
            try:
                self.weights, self.bias = self._load_params()
                print(colored(f"Succesfully Loaded Save Parameters from {self.load_params}\n", 'green', attrs=['bold']))
            except:
                print(colored(f"WARNING - {self.load_params} - PARAMS NOT FOUND. Specify proper file path or set load_params to False.\nHit CTRL + C to stop training run, if desired.\n", 'red', attrs=['bold']))
                print(colored(f"Initializing New Parameters\n", 'green', attrs = ['bold']))
                for layer, neurons in list(self.nn_capacity.items())[1:]:
                    self.weights[layer - 1] = jx.random.normal(self.__key, shape = (neurons, list(self.nn_capacity.values())[layer - 1])) * np.sqrt(2 / list(self.nn_capacity.values())[layer - 1]) 
                    self.bias[layer - 1] = np.zeros(shape = (neurons, 1))
        else:
            for layer, neurons in list(self.nn_capacity.items())[1:]:
                self.weights[layer - 1] = jx.random.normal(self.__key, shape = (neurons, list(self.nn_capacity.values())[layer - 1])) * np.sqrt(2 / list(self.nn_capacity.values())[layer - 1]) 
                self.bias[layer - 1] = np.zeros(shape = (neurons, 1))
        
    def _gradient_descent(self):
              
        self._init_training_logs() 
        
        start_time = t.time()
        for epoch in range(self.epochs):
            self.c_epoch = epoch + 1
            self._forward()
            if self.train_verbose:
                print(f"EPOCH -- {epoch + 1} | ACCURACY -- {self._accuracy()}% | CROSS ENTROPY LOSS -- {self._cross_entropy()}") 
            self._backward() 
            self._update_params()
        end_time = t.time()

        print(f"\nTotal Training Time: {end_time - start_time}\n") 
     
    def _forward(self): 
        
        for layer in self.__layers_idxs:
            if layer == 1: 
                self.preactivations[layer - 1] = np.dot(self.weights[layer - 1], self.X_train) + self.bias[layer - 1]
            else:
                self.preactivations[layer - 1] = np.dot(self.weights[layer - 1], self.activations[layer - 2]) + self.bias[layer - 1]
                
            if layer != len(self.__layers_idxs):
                self.activations[layer - 1] = self._Leaky_ReLU(self.preactivations[layer - 1])
            else:
                self.activations[layer - 1] = self._softmax(self.preactivations[layer - 1]) 

    def _backward(self):
        for layer in reversed(self.__layers_idxs):
            if layer == self.__layers_idxs[-1]:
                self.grad_preacts[layer - 1] = self.activations[layer - 1] - self.Y_onehot
                self.grad_weights[layer - 1] = np.dot(self.grad_preacts[layer - 1], self.activations[layer - 2].T) * ( 1 / self.Y_train.size )
                self.grad_bias[layer - 1] = np.sum(self.grad_preacts[layer - 1], axis = 1, keepdims = True) * ( 1 / self.Y_train.size )
               
                self.v_grad_weights[layer - 1] = ((self.beta_1 * self.v_grad_weights[layer - 1]) + ((1- self.beta_1) * self.grad_weights[layer-1]))
                self.v_grad_bias[layer - 1] = (self.beta_1 * self.v_grad_bias[layer - 1]) + ((1 - self.beta_1) * self.grad_bias[layer - 1]) 
               
                self.s_grad_weights[layer - 1] = ((self.beta_2 * self.s_grad_weights[layer - 1]) + ((1 - self.beta_2) * np.square(self.grad_weights[layer - 1])))
                self.s_grad_bias[layer - 1]  = ((self.beta_2 * self.s_grad_bias[layer - 1]) + (( 1 - self.beta_2) * np.square(self.grad_bias[layer - 1])))
                
            elif layer not in [self.__layers_idxs[-1], self.__layers_idxs[0]]:   
                self.grad_preacts[layer - 1] = np.dot(self.weights[layer].T, self.grad_preacts[layer]) * self._grad_Leaky_ReLU(self.preactivations[layer - 1]) 
                self.grad_weights[layer - 1] = np.dot(self.grad_preacts[layer - 1], self.activations[layer - 2].T)  * ( 1 / self.Y_train.size )
                self.grad_bias[layer - 1] = np.sum(self.grad_preacts[layer - 1], axis = 1, keepdims = True) * ( 1 / self.Y_train.size )
                
                self.v_grad_weights[layer - 1] = ((self.beta_1 * self.v_grad_weights[layer - 1]) + ((1 - self.beta_1 ) * self.grad_weights[layer-1])) 
                self.v_grad_bias[layer - 1] = ((self.beta_1 * self.v_grad_bias[layer - 1]) + ((1 - self.beta_1 ) * self.grad_bias[layer - 1])) 
                
                self.s_grad_weights[layer - 1] = ((self.beta_2 * self.s_grad_weights[layer - 1]) + ((1 - self.beta_2) * np.square(self.grad_weights[layer - 1])))
                self.s_grad_bias[layer - 1]  = ((self.beta_2 * self.s_grad_bias[layer - 1]) + (( 1 - self.beta_2) * np.square(self.grad_bias[layer - 1])))
                
            else:
                self.grad_preacts[layer - 1] = np.dot(self.weights[layer].T, self.grad_preacts[layer]) * self._grad_Leaky_ReLU(self.preactivations[layer - 1])
                self.grad_weights[layer - 1] = np.dot(self.grad_preacts[layer - 1], self.X_train.T)  * ( 1 / self.Y_train.size )
                self.grad_bias[layer - 1] = np.sum(self.grad_preacts[layer - 1], axis = 1, keepdims = True) * ( 1 / self.Y_train.size )
                
                self.v_grad_weights[layer - 1] = ((self.beta_1 * self.v_grad_weights[layer - 1]) + ((1 - self.beta_1 ) * self.grad_weights[layer-1]))
                self.v_grad_bias[layer - 1] = ((self.beta_1 * self.v_grad_bias[layer - 1]) + ((1 - self.beta_1) * self.grad_bias[layer - 1])) 
                
                self.s_grad_weights[layer - 1] = ((self.beta_2 * self.s_grad_weights[layer - 1]) + ((1 - self.beta_2) * np.square(self.grad_weights[layer - 1])))
                self.s_grad_bias[layer - 1]  = ((self.beta_2 * self.s_grad_bias[layer - 1]) + (( 1 - self.beta_2) * np.square(self.grad_bias[layer - 1])))
                
    def _update_params(self):
        for layer in reversed(self.__layers_idxs):
            self.weights[layer - 1] -= (self.alpha / np.sqrt(self.s_grad_weights[layer - 1]) + self.eps) * self.v_grad_weights[layer - 1]
            self.bias[layer - 1] -= (self.alpha / np.sqrt(self.s_grad_bias[layer - 1]) + self.eps) * self.v_grad_bias[layer - 1]

    def _inference(self): 

        for layer in self.__layers_idxs:
            if layer == 1: 
                self.preactivations[layer - 1] = np.dot(self.weights[layer - 1], self.X_test) + self.bias[layer - 1]
            else:
                self.preactivations[layer - 1] = np.dot(self.weights[layer - 1], self.activations[layer - 2]) + self.bias[layer - 1]
                
            if layer != len(self.__layers_idxs):
                self.activations[layer - 1] = self._Leaky_ReLU(self.preactivations[layer - 1])
            else:
                self.activations[layer - 1] = self._softmax(self.preactivations[layer - 1]) 

        return self.activations[-1]


    def _Leaky_ReLU(self, z):
        return np.maximum(.01 * z, z)  
   
    def _grad_Leaky_ReLU(self, z):
        return np.where(z > 0, 1, .01 )
    
    def _softmax(self, z):
        z = z - np.max(z, axis = 0, keepdims = True)
        return np.exp(z + self.eps) / np.sum(np.exp(z + self.eps), axis = 0, keepdims = True) 
   
    def _cross_entropy(self):
        return - np.sum(self.Y_onehot * np.log(self.activations[-1] + self.eps)) * ( 1 / self.Y_train.size)

    def _test_cross_entropy(self):
        return - np.sum(self.Y_onehot * np.log(self.activations[-1] + self.eps)) * ( 1 / self.Y_test.size)

    def _accuracy(self):
        return np.mean(self.Y_train.flatten() == np.argmax(self.activations[-1], axis = 0)) * 100 # 10, 60000
   
    def _test_accuracy(self):
        return np.mean(self.Y_test.flatten() == np.argmax(self.activations[-1], axis = 0 )) * 100
    
    def _get_params(self):
        
        try:
            self.weights, self.bias, self.__layers_idxs = self._load_params()
            print(colored(f"Parameters Found at {self.load_params}. Beginning Testing.\n", 'green', attrs = ['bold']))
        except: 
            if hasattr(self, 'weights') and hasattr(self, 'bias'):
                print(colored(f"Parameters Found at Neural Network Instance Object. Beginning Testing.", 'green', attrs = ['bold']))
            else:
                raise ValueError(colored(f"ERROR - Parameters not Found. Train your model and test within the same instance. Or Load parameters via `load_params` filepath argument.", color = 'red', attrs = ['bold']))


    def _init_internals(self):
        
        self.__layers_idxs = list(self.nn_capacity.keys())[1:]
        
        self.preactivations = [0 * l for l in self.__layers_idxs]
        self.activations = [0 * l for l in self.__layers_idxs] 
        self.weights = [0 * l for l in self.__layers_idxs] 
        self.bias = [0 * l for l in self.__layers_idxs] 
        self.grad_preacts = [0 * l for l in self.__layers_idxs]
        self.grad_weights = [0 * l for l in self.__layers_idxs]
        self.grad_bias = [0 * l for l in self.__layers_idxs]
        self.v_grad_weights = [0 * l for l in self.__layers_idxs]
        self.v_grad_bias = [0 * l for l in self.__layers_idxs]
        self.s_grad_weights = [0 * l for l in self.__layers_idxs]
        self.s_grad_bias = [0 * l for l in self.__layers_idxs]
        
    def _init_training_logs(self):
        print(colored(f"Training a Neural Network of size:\n", 'green', attrs = ['bold']))
        for i in self.__layers_idxs:
            if i == self.__layers_idxs[-1]:
                print(colored(f"Output Layer ({i}): {list(self.nn_capacity.values())[i]} Output Units", 'green', attrs = ['bold']))
            else:
                print(colored(f"Layer {i}: {list(self.nn_capacity.values())[i]} Hidden Units", 'green', attrs = ['bold'])) 
        print(colored(f'\nBeginning Training, for {self.epochs} epochs, in 3 seconds.\n', 'green', attrs = ['bold']))
        for i in range(3):
            print(colored(i, 'green', attrs=['bold']))
            t.sleep(1)
        print() 

    def _save_params(self):
        if self.save_params:
            with open(self.save_params, 'wb') as f:
                pkl.dump((self.weights, self.bias, self.__layers_idxs), f)
    
    def _load_params(self):
        with open(self.load_params, 'rb') as f:
            return pkl.load((f)) 
           
    def _debug(self):
        print()
        print(f"Output grad_preacts: {self.grad_preacts[1].shape} | Output preacts: {self.preactivations[1].shape}")
        print(f"Output grad_weights: {self.grad_weights[1].shape} | output weights: {self.weights[1].shape}")
        print(f"Output grad_bias: {self.bias[1].shape} | output bias: {self.bias[1].shape}")
        print()
        print(f"Hidden grad_preacts: {self.grad_preacts[0].shape} | Hidden preacts: {self.preactivations[0].shape}")
        print(f"Hidden grad_weights: {self.grad_weights[0].shape} | Hidden weights: {self.weights[0].shape}")
        print(f"Hidden grad_bias: {self.bias[0].shape} | Hidden bias: {self.bias[0].shape}")

    @property
    def Y_onehot(self):
        return self._Y_onehot

    @Y_onehot.setter
    def Y_onehot(self, Y_onehot):
        if Y_onehot is None:
            print(colored("No One Hot Encoding found. One Hot Encoding Training Labels...", "green", attrs = ['bold']))
            Y_onehot = jnn.one_hot(self.Y_train, num_classes = np.max(self.Y_train) + 1, axis = 0)
            if len(Y_onehot.shape) == 3:
                Y_onehot = np.squeeze(Y_onehot, axis = 2)
        self._Y_onehot = Y_onehot

    @property
    def Y_test_onehot(self):
        return self._Y_test_onehot
    
    @Y_test_onehot.setter
    def Y_test_onehot(self, Y_test_onehot):
        if Y_test_onehot is None:
            print(colored("No One Hot Encoding found. One Hot Encoding Testing Labels...", "green", attrs = ['bold']))
            Y_test_onehot = jnn.one_hot(self.Y_test, num_classes = np.max(self.Y_test) + 1, axis = 0)
            if len(Y_test_onehot.shape) == 3:
                Y_test_onehot = np.squeeze(Y_test_onehot, axis = 2)
        self._Y_test_onehot = Y_test_onehot

    @property
    def seed(self):
        return self._seed
    
    @seed.setter
    def seed(self, seed):
        if seed is None:
            self._seed = random.randint(0, 1000)
        else:
            self._seed = seed
        self.__key = jx.random.key(self._seed)

if __name__ == "__main__":
    
    nnet = NN(train_verbose = True, seed = 1)
    
    train_data = csv_to_numpy('data/fashion-mnist_train.csv')
    X_train, Y_train = x_y_split(train_data, y_col = 'first') 
    X_train = X_train.T / 255

    epochs = 100
    alpha = .01
    beta_1 = .99
    beta_2 = .99
    
    save_params = 'models/adamNN.pkl'
    load_params = 'models/adamNN.pkl'
    
    nn_capacity = {
       0: 784, 
       1: 32,
       2: 10
    }
    
    nnet.train(X_train, Y_train, epochs = epochs, alpha = alpha, beta_1 = beta_1, beta_2 = beta_2, nn_capacity=nn_capacity, save_params = save_params, load_params = load_params)
    
    test_data = csv_to_numpy('data/fashion-mnist_train.csv')
    X_test, Y_test = x_y_split(test_data, y_col = 'first')
    X_test = X_test.T / 255
   
    nnet.test(X_test, Y_test, load_params = load_params)