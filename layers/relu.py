import numpy as np

class RELU_LAYER:
    """docstring forRELU_LAYER."""
    def __init__(self):
        pass

    def forward(self, X):
        """
        Computes the forward pass of Relu Layer.
        Input:
            X: Input data of any shape
        """
        """
        Dokumentasi
        
        input : 
        output :
        
        """
        self.cache = X
        self.output = np.maximum(X, 0)
        return self.output, 0

    def backward(self, delta):
        """
        Computes the backward pass of Relu Layer.
        Input:
            delta: Shape of delta values should be same as of X in cache
        """
        """
         Zero gradient where input values were negative
         we just copy value from next layer because 
         derivative of relu if z > 0 = 1, else 0
         in chain rule we will multiply dvalue by one 
         we dont have to do the multiply, just copy the dvalue
         so we just have to make sure to change negative value to 0
         drelu_dz = dvalue * (1. if z > 0 else 0.)
        """
        """
        Dokumentasi
        
        input : 
        output :
        
        """
        self.delta_X = delta.copy()
        self.delta_X[self.cache <=0] = 0
        #self.delta_X = delta * (self.cache >= 0)
        return self.delta_X
