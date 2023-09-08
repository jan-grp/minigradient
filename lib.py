import random
from main import Mini

random.seed(42)

class Neuron():
    def __init__(self, num_weights, use_activation_function=True):
        self.weights = [Mini(random.uniform(-1, 1)) for _ in range(num_weights)]
        self.bias = Mini(0)
        self.use_activation = use_activation_function

    def __call__(self, X):
        assert len(self.weights) == len(X), f"Input/weight mismatch; number of weights: {len(self.weights)}, number of inputs: {len(X)}"
        result = sum(w*x for w, x in zip(self.weights, X)) + self.bias
        return result.sigmoid() if self.use_activation else result
    
    def params(self): # used for optimization (training)
        return self.weights + [self.bias]
    
    def reset_gradients(self):
        for param in self.params():
            param.reset_gradient()

class Layer():
    def __init__(self, num_neurons, num_weights_per_neuron, use_activatino_function=True):
        self.neurons = [Neuron(num_weights=num_weights_per_neuron,
                               use_activation_function=use_activatino_function) for _ in range(num_neurons)]
        
    def __call__(self, X):
        outputs = [neuron(X=X) for neuron in self.neurons]
        return outputs
    
    def params(self): # used for optimization (training)
        return [parameter for neuron in self.neurons for parameter in neuron.params()]
    
    def reset_gradients(self): # used after each optimization step
        for param in self.params():
            param.reset_gradient()
    