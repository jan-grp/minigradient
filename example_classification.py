from minigradient.lib import Layer, plot_loss
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# dataset
X, Y = make_moons(n_samples=100, noise=0.14)
Y = Y*2 - 1

def plot_dataset(X, Y):
    # 2 D representation of the data
    plt.figure(figsize=(5,5))
    plt.scatter(X[:,0], X[:,1], c=Y, s=20)
    plt.show()
# plot_dataset(X, Y)

X_train, Y_train = np.array(X).transpose(), np.array(Y).reshape((1, len(Y)))

# model 
n_x = X_train.shape[0]
n_y = Y_train.shape[0]

l1 = Layer(num_neurons=2, num_weights_per_neuron=n_x, use_activatino_function=True)
l2 = Layer(num_neurons=3, num_weights_per_neuron=2, use_activatino_function=True)
final_layer = Layer(num_neurons=1, num_weights_per_neuron=3, use_activatino_function=True)

model_params = np.concatenate((l1.params(), l2.params(), final_layer.params()))

def predict(X):
    l1_out = l1(X)
    l2_out = l2(l1_out)
    final_out = final_layer(l2_out)
    return final_out

# training parameters
num_iterations = 1
learning_rate = 1

# training
loss_array = []

for _ in range(num_iterations):
    # make predictions on the hole dataset
    predictions = [predict([x1, x2]) for x1, x2 in zip(X_train[0], X_train[1])]

    # loss of each prediction
    losses = [-(y*y_hat.log() + (1 - y)*(1 - y_hat).log()) for y, y_hat in zip(Y_train[0], predictions)]

    # calculate the average loss
    avg_loss = sum(losses) / len(losses)
    loss_array.append(avg_loss.value)

    # optimization
    avg_loss.backprop()
    for param in model_params:
        print("initial value: ", param.value)
        print("gradient: ", param.gradient)
        param.value = param.value - learning_rate*param.gradient
        print("updated value: ", param.value)
        param.reset_gradient()
        break

# plot_loss(loss_array)
print("initial loss: ", loss_array[0])
print("final loss: ", loss_array[-1])