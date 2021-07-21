import numpy as np
import sys
class FC_LAYER:
    """docstring forRELU_LAYER."""
    def __init__(self, layer_size, kernel_size, **params):
        """
        Input:
            layer_size: number of neurons/nodes in fc layer
            kernel: kernel of shape (nodes_l1 , nodes_l2) (input output)
            fan: tuple of number of nodes in previous layer and this layer
        """
        """
        Dokumentasi
        
        input : 
        output :
        
        """
        self.nodes = layer_size

        fname = params.get("filename", None)
        if fname:
            try:
                arr_files = np.load(fname)
                self.kernel = arr_files['arr_0']
                self.bias = arr_files['arr_1']
                assert np.all(self.kernel.shape == kernel_size) and np.all(self.bias.shape[0] == kernel_size[1])
            except:
                 raise
        else:
            f = np.sqrt(6)/np.sqrt(kernel_size[0] + kernel_size[1])
            epsilon = 1e-6
            self.kernel = np.random.uniform(-f, f + epsilon, kernel_size)
            self.bias = np.random.uniform(-f, f + epsilon, (1,kernel_size[1]))
        
        pass

    def load(self,path,kernel_size):
        """
        Dokumentasi
        
        input : 
        output :
        
        """
        if path:
            try:
                arr_files = np.load(path)
                self.kernel = arr_files['arr_0']
                self.bias = arr_files['arr_1']
                #print("\nlayer name {} :: kernel and bias shape {} {} :: kernel size {}".format(self.layer_name,self.kernel.shape,self.bias.shape,kernel_size))
                assert np.all(self.kernel.shape == kernel_size) and np.all(self.bias.shape[1] == kernel_size[1])
            except:
                 raise
        
    
    def forward(self, X):
        """
        Computes the forward pass of Sigmoid Layer.
        Input:
            X: Input data of shape (N, nodes_l1)
        Variables:
            kernel: Weight array of shape (nodes_l1, nodes_l2)
            bias: Biases of shape (nodes_l2)
        where,
            nodes_l1: number of nodes in previous layer
            nodes_l2: number of nodes in this fc layer
        """
        """
        Dokumentasi
        
        input : 
        output :
        
        """
        #kernel, bias = self.kernel, self.bias
        self.cache = X
        self.activations = np.dot(X, self.kernel) + self.bias
        #assert self.activations.shape == (X.shape[0], bias.shape[0])
        return self.activations, np.sum(np.square(self.kernel))

    def backward(self, delta):
        """
        Computes the backward pass of Sigmoid Layer.
        Input:
            delta: Shape of delta values (N, nodes_l2)
        """
        X = self.cache
        self.delta_X = np.dot(delta, self.kernel.T)
        self.delta_K = np.dot(X.T, delta)
        #print(self.delta_K[0][range(10)], self.delta_K.shape)
        #print(X.T[0][range(10)], X.T.shape)
        #print(delta[0][range(10)], delta.shape)
        self.delta_b = np.sum(delta, axis=0)
        return self.delta_X

    def update_kernel(self, **params):
        """
        Update kernel and biases stored in this layer.
        Input:
            params: Optional parameters- method, alpha, zeta
        """
        """
        Dokumentasi
        
        input : 
        output :
        
        """
        method       = params.get("method", "")
        alpha        = params.get("alpha", 0.001)
        mu           = params.get("mu", 0.9) #momentum factor
        zeta         = params.get("zeta", 0.01)
        beta1        = params.get("beta1", 0.9)
        beta2        = params.get("beta2", 0.999)
        batch_size   = params.get("batch", 1)
        fudge_factor = 1e-8                         # smoothing term to avoid division by zero
        
        if not hasattr(self,'gradient_history'):
            temp = self.kernel.shape
            self.gradient_history = np.zeros((self.kernel.shape))
            self.bias_history = np.zeros(temp[1])
            self.m_kernel = np.zeros((self.kernel.shape))
            self.m_bias = np.zeros(temp[1])
            self.v_kernel = np.zeros((self.kernel.shape))
            self.v_bias = np.zeros(temp[1])
            self.timestamp = 0
        
        if method == "adagrad":
            self.gradient_history += np.square(self.delta_K + (zeta*self.kernel/batch_size))
            self.bias_history += np.square(self.delta_b)
            #print("\n\n\n\n\n\n")
            #print(self.gradient_history[0])
            #sys.exit()
            #print(alpha*(self.delta_K + (zeta*self.kernel/batch_size))/(np.sqrt(self.gradient_history) + fudge_factor))
            self.kernel -= np.divide(alpha*(self.delta_K + (zeta*self.kernel/batch_size)), (np.sqrt(self.gradient_history) + fudge_factor))
            self.bias -= np.divide(alpha*self.delta_b, (np.sqrt(self.bias_history) + fudge_factor))
        elif method == "gd_momentum":
            new_delta_K = alpha*(self.delta_K + (zeta*self.kernel/batch_size)) + mu*self.gradient_history
            new_delta_b = alpha*self.delta_b + mu*self.bias_history
            self.kernel -= new_delta_K
            self.bias -= new_delta_b
            self.gradient_history = self.delta_K + (zeta*self.kernel/batch_size)
            self.bias_history = self.delta_b
        elif method == "adam":
            self.timestamp += 1
            alpha = alpha * np.sqrt(1 - np.power(beta2, self.timestamp)) / (1 - np.power(beta1, self.timestamp))
            self.m_kernel = beta1 * self.m_kernel + (1 - beta1) * (self.delta_K + (zeta*self.kernel/batch_size))
            self.m_bias = beta1 * self.m_bias + (1 - beta1) * self.delta_b
            self.v_kernel = beta2 * self.v_kernel + (1 - beta2) * np.square((self.delta_K + (zeta*self.kernel/batch_size)))
            self.v_bias = beta2 * self.v_bias + (1 - beta2) * np.square(self.delta_b)

            self.kernel -= np.divide(alpha * self.m_kernel, (np.sqrt(self.v_kernel) + fudge_factor))
            self.bias -= np.divide(alpha * self.m_bias, (np.sqrt(self.v_bias) + fudge_factor))
        else:
            self.kernel -= alpha*(self.delta_K + zeta*self.kernel/batch_size)
            self.bias    -= alpha*self.delta_b
        pass
