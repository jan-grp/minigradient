import random
from minigradient.main import Mini
import matplotlib.pyplot as plt

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
        outputs = [neuron(X=X) for neuron in self.neurons] if len(self.neurons) > 1 else self.neurons[0](X)
        return outputs
    
    def params(self): # used for optimization (training)
        return [parameter for neuron in self.neurons for parameter in neuron.params()]
    
    def reset_gradients(self): # used after each optimization step
        for param in self.params():
            param.reset_gradient()
    

def plot_loss(loss_array):
    _, ax = plt.subplots()
    ax.plot(loss_array)
    ax.set_xlabel('training iterations')
    ax.set_ylabel('loss')
    ax.set_title(f'loss over training iterations (final loss: {round(loss_array[-1], 4)})')

    plt.show()