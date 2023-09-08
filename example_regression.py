# train a model to convert Celcius to Fahrenheit

from minigradient.lib import Neuron
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# construct data
X = np.random.randn(100)
Y = np.array([1.8*x+32 for x in X])

X_norm = (X - np.mean(X))/np.std(X)
Y_norm = (Y - np.mean(Y))/np.std(Y)

# model
model = Neuron(num_weights=1, use_activation_function=False)
parameters = model.params()

# training parameters
num_iterations = 2000
learning_rate = 0.1

# training
loss_array = []
for _ in range(num_iterations):
    # make predictions on the hole dataset
    predictions = [model([x]) for x in X_norm]

    # calculate the loss for each prediction
    losses = [(y - y_hat)**2 / 2 for y, y_hat in zip(Y_norm, predictions)]

    # calculate the average loss
    avg_loss = sum(losses) / len(losses)
    loss_array.append(avg_loss.value)
    
    # update parameters (gradient descent)
    avg_loss.backprop()
    for param in parameters:
        param.value = param.value - learning_rate*param.gradient

    # reset the gradients for the next training step
    model.reset_gradients()

# summerize the results
print("Summary:\n")
print(f'num_iterations: {num_iterations}\nlearning_rate: {learning_rate}\nfinal_loss: {loss_array[-1]}')
print(f'weight: {parameters[0].value}\nbias: {parameters[1].value}')

predictions_norm = [model([x]) for x in X_norm[:3]]
# Denormalize the predictions to get the final predictions
predictions_denormalized = [p.value * np.std(Y) + np.mean(Y) for p in predictions_norm]

print("\nMake predictions:\n")
print(f'input: {X[:3]}\ndesired output: {Y[:3]}\nactual output: {predictions_denormalized}')

def plot_loss(loss_array):
    _, ax = plt.subplots()
    ax.plot(loss_array)
    ax.set_xlabel('training iterations')
    ax.set_ylabel('loss')
    ax.set_title(f'loss over training iterations (final loss: {round(loss_array[-1], 4)})')

    plt.show()

plot_loss(loss_array)