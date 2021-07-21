import numpy as np
#Adagrad Optimizer
class Optimizer_Adagrad:

    """
        Dokumentasi
        
        input : 
        output :
        
    """
    # Initialize optimizer - set settings
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        # params
        # learning_rate = fixed learning rate
        # current_learning_rate = dinamic learning rate, learning rate decreased 
        # epsilon param prevent zero division
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    # Call once before any parameter updates
    def pre_update_params(self):
        """
        Dokumentasi
        
        input : 
        output :
        
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        """
        Dokumentasi
        
        input : 
        output :
        
        """
        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'kernel_cache'):
            layer.kernel_cache = np.zeros_like(layer.kernel)
            layer.bias_cache = np.zeros_like(layer.bias)

        # Update cache with squared current gradients
        # The cache holds a history of squared gradients
        layer.kernel_cache += layer.delta_K**2
        layer.bias_cache += layer.delta_b**2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        # \ or backslash used as break in python
        layer.kernel += -self.current_learning_rate * \
                         layer.delta_K / \
                         (np.sqrt(layer.kernel_cache) + self.epsilon)
        layer.bias += -self.current_learning_rate * \
                        layer.delta_b / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

# SGD optimizer
class Optimizer_SGD:
    """
    Dokumentasi
        
    input : 
    output :
        
    """
    # Initialize optimizer - set settings, 
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=0.001, decay=0.0, momentum=0.9):
        """
        Dokumentasi
        
        input : 
        output :
        
        """
        
        # params
        # learning_rate = fixed learning rate
        # current_learning_rate = dinamic learning rate, learning rate decreased each epoch
        # momentums value between 0 and 1
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # Call once before any parameter updates
    def pre_update_params(self):
        """
        Dokumentasi
        
        input : 
        output :
        
        """
        # if used, vlue wont be 0
        # decaying learning rate is about decreasing learning rate each epoch 
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        """
        Dokumentasi
        
        input : 
        output :
        
        """
        # If we use momentum
        # if used, vlue wont be 0
        # momentum uses the previous update’s direction to influence the next update’s direction
        if self.momentum:

            # If layer does not contain momentum arrays, create them
            # filled with zeros
            if not hasattr(layer, 'weight_history'):
                layer.weight_history = np.zeros_like(layer.kernel)
                # If there is no momentum array for weights
                # make momentums attribute
                # layer momentums start form zero it means no initial directions

                # The array doesn't exist for biases yet either.
                layer.bias_history = np.zeros_like(layer.bias)

            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            # \ or backslash used as break in python
            weight_updates = (self.momentum * layer.weight_history) + ((1 - self.momentum) * layer.delta_K)
            """
            weight_updates = \
                self.momentum * layer.weight_history - \
                self.current_learning_rate * layer.delta_K
            """
            # update layer weight momentums directions
            layer.weight_history = weight_updates

            # Build bias updates
            bias_updates = (self.momentum * layer.bias_history) + ((1 - self.momentum) * layer.delta_b)
            """
            bias_updates = \
                self.momentum * layer.bias_history - \
                self.current_learning_rate * layer.delta_b
            """
            # update layer bias momentums directions
            layer.bias_history = bias_updates

        # Vanilla SGD updates (as before momentum update)
        else:
            weight_updates = -self.current_learning_rate * \
                             layer.delta_K
            bias_updates = -self.current_learning_rate * \
                           layer.delta_b

        # Update weights and biases using either
        # vanilla or momentum updates
        layer.kernel -= (self.current_learning_rate * weight_updates)
        layer.bias -= ( self.current_learning_rate * bias_updates)

    # Call once after any parameter updates
    # marked the iteration position
    def post_update_params(self):
        self.iterations += 1

# RMSprop optimizer
class Optimizer_RMSprop:
    """
    Dokumentasi
        
    input : 
    output :
        
    """
    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        """
        Dokumentasi
        
        input : 
        output :
        
        """
        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.kernel)
            layer.bias_cache = np.zeros_like(layer.bias)

        # The cache holds a history of squared gradients
        # Update cache with squared current gradients
        layer.weight_cache = np.add(np.multiply(self.rho, layer.weight_cache), \
                                    np.multiply((1 - self.rho), np.power(layer.delta_K,2)))
        layer.bias_cache = np.add(np.multiply(self.rho, layer.bias_cache), \
                                  np.multiply((1 - self.rho), np.power(layer.delta_b, 2)))
        """
        self.rho * layer.bias_cache + \
            (1 - self.rho) * layer.delta_b**2
        """

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        # \ or backslash used as break in python
        layer.kernel = np.add(layer.kernel, np.divide(np.multiply(np.negative(self.current_learning_rate), layer.delta_K), \
                                                      np.add(np.sqrt(layer.weight_cache), self.epsilon)))
        layer.bias = np.add(layer.bias, np.divide(np.multiply(np.negative(self.current_learning_rate), layer.delta_b), \
                                                  np.add(np.sqrt(layer.bias_cache), self.epsilon)))
        
        """
        layer.kernel += -self.current_learning_rate * \
                         layer.delta_K / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.bias += -self.current_learning_rate * \
                        layer.delta_b / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)
        """
        

    # Call once after any parameter updates
    def post_update_params(self):
        """
        Dokumentasi
        
        input : 
        output :
        
        """
        self.iterations += 1

class Optimizer_Adam:
    """
    Dokumentasi
        
    input : 
    output :
        
    """
    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update_params(self):
        """
        Dokumentasi
        
        input : 
        output :
        
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        """
        Dokumentasi
        
        input : 
        output :
        
        """
        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.kernel)
            layer.weight_cache = np.zeros_like(layer.kernel)
            layer.bias_momentums = np.zeros_like(layer.bias)
            layer.bias_cache = np.zeros_like(layer.bias)

        # Update momentum  with current gradients
        """
        layer.weight_momentums = self.beta_1 * \
                                 layer.weight_momentums + \
                                 (1 - self.beta_1) * layer.delta_K
        layer.bias_momentums = self.beta_1 * \
                               layer.bias_momentums + \
                               (1 - self.beta_1) * layer.delta_b
        """ 
        layer.weight_momentums = np.add(np.multiply(self.beta_1, layer.weight_momentums), \
                                        np.multiply((1 - self.beta_1), layer.delta_K))
        layer.bias_momentums = np.add(np.multiply(self.beta_1, layer.bias_momentums), \
                                      np.multiply((1 - self.beta_1), layer.delta_b))
        
        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = np.divide(layer.weight_momentums,\
            (1- np.power(self.beta_1, (self.iterations + 1))))
        bias_momentums_corrected = np.divide(layer.bias_momentums, \
            (1- np.power(self.beta_1, (self.iterations + 1))))
        
        # Update cache with squared current gradients
        
        layer.weight_cache = np.add(np.multiply(self.beta_2, layer.weight_cache), \
            np.multiply((1 - self.beta_2), np.power(layer.delta_K,2)))
        layer.bias_cache = np.add(np.multiply(self.beta_2, layer.bias_cache), \
            np.multiply((1 - self.beta_2), np.power(layer.delta_b,2)))
        
        
        # Get corrected cache
        weight_cache_corrected = np.divide(layer.weight_cache, \
            (1 - np.power(self.beta_2, (self.iterations + 1))))
        bias_cache_corrected = np.divide(layer.bias_cache, \
            (1 - np.power(self.beta_2, (self.iterations + 1))))
        
        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        
        layer.kernel = np.add(layer.kernel, np.multiply(-self.current_learning_rate, \
                         np.divide(weight_momentums_corrected, \
                         (np.sqrt(weight_cache_corrected) + self.epsilon))))
                         
        layer.bias = np.add(layer.bias, np.multiply(-self.current_learning_rate, \
                         np.divide(bias_momentums_corrected, \
                         (np.sqrt(bias_cache_corrected) + self.epsilon)))
 )
    # Call once after any parameter updates
    def post_update_params(self):
        """
        Dokumentasi
        
        input : 
        output :
        
        """
        self.iterations += 1
