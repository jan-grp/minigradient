from main import Mini
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1111)

# learn to convert Celcius to Fahrenheit

# construct data
X = np.random.randn(100)
Y = np.array([1.8*x+32 for x in X])

X_norm = (X - np.mean(X))/np.std(X)
Y_norm = (Y - np.mean(Y))/np.std(Y)

num_iterations = 2000
learning_rate = 0.1
w = Mini((np.random.randn(1) * 0.1)[0])
b = Mini(0)

loss_array = []
for _ in range(num_iterations):
    predictions = [w*x + b for x in X_norm]
    losses = [(y - y_hat)**2 / 2 for y, y_hat in zip(Y_norm, predictions)]
    avg_loss = sum(losses) / len(losses)
    loss_array.append(avg_loss.value)
    avg_loss.backprop()
    w = Mini(w.value - learning_rate*w.gradient)
    b = Mini(b.value - learning_rate*b.gradient)

print("Summary:\n")
print(f'num_iterations: {num_iterations}\nlearning_rate: {learning_rate}\nfinal_loss: {loss_array[-1]}')
print(f'weight: {w.value}\nbias: {b.value}')

predictions_norm = [w*x + b for x in X_norm[:3]]
# Denormalize the predictions to get the final predictions
predictions_unnormalized = [p.value * np.std(Y) + np.mean(Y) for p in predictions_norm]

print("\nMake predictions:\n")
print(f'input: {X[:3]}\ndesired output: {Y[:3]}\nactual output: {predictions_unnormalized}')

def plot_loss(loss_array):
    _, ax = plt.subplots()
    ax.plot(loss_array)
    ax.set_xlabel('training iterations')
    ax.set_ylabel('loss')
    ax.set_title(f'loss over training iterations (final loss: {round(loss_array[-1], 4)})')

    plt.show()

plot_loss(loss_array)