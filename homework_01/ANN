from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
import random

# Load the digits dataset
digits = load_digits()

# Extract the data into (input, target) tuples
data = list(zip(digits.data, digits.target))

# Convert each image to float32
float32_data = [(image.astype(np.float32) / 255, label) for image, label in data]
new_data = tuple(float32_data)

targets = digits.target

universe = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

mapping = {}
for x in range(len(universe)):
    mapping[universe[x]] = x

onehotencode = []
for i in targets:
    arr = list(np.zeros(len(universe), dtype=int))
    arr[mapping[i]] = 1
    onehotencode.append(arr)

minibatch_size = 10


def shuffle_data(data, minibatch_size):
    indices = list(range(len(data)))
    random.shuffle(indices)
    for i in range(0, len(indices), minibatch_size):
        excerpt = indices[i:i + minibatch_size]
        minibatch_inputs = np.array([data[j][0] for j in excerpt])
        minibatch_targets = np.array([onehotencode[j] for j in excerpt])

        yield minibatch_inputs, minibatch_targets


class Sigmoid:
    def __init__(self):
        self.activation = None

    def sigmoid(self, x):
        self.activation = 1 / (1 + np.exp(-x))
        return self.activation

    def sigmoid_derivative(self, x):
        return self.activation * (1 - self.activation)

    def forward(self, x):
        self.activation = self.sigmoid(x)
        return self.activation

    def backward(self, pre_activation, error_signal):
        # Compute the derivative of the loss with respect to the pre-activation
        sigmoid_derivative = self.sigmoid_derivative(pre_activation)
        loss_derivative = error_signal * sigmoid_derivative

        return loss_derivative


class SoftmaxActivation:
    def __init__(self):
        self.output = None

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output

    def backward(self, gradient):
        # This is the backward pass for softmax
        # 'gradient' here is the gradient of the loss with respect to the output of the softmax

        # Calculate Jacobian matrix for softmax
        jacobian = np.diag(self.output[0]) - np.outer(self.output[0], self.output[0])

        # Backpropagate the gradient through the softmax layer
        return gradient @ jacobian


class MLP_layer:
    def __init__(self, input_size, output_size, learning_rate=0.01, low=0.3, high=0.3):
        self.weights = np.random.uniform(low=low, high=high, size=(input_size, output_size))
        self.bias = np.zeros((1, output_size))
        self.pre_activation = None
        self.learning_rate = learning_rate  # Add learning_rate parameter

    def call(self, x):
        self.pre_activation = x @ self.weights + self.bias
        return self.pre_activation

    def backward(self, x, gradient):
        dw = x.T @ gradient
        db = np.sum(gradient, axis=0, keepdims=True)
        dx = gradient @ self.weights.T

        # Update weights and bias using learning rate
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

        return dx


class MLP:
    def __init__(self, input_size, hidden_layer_sizes, output_layer_size, learning_rate=0.01):
        self.hidden_layers = []
        for layer_index in range(len(hidden_layer_sizes)):
            this_layer_input_size = hidden_layer_sizes[layer_index - 1] if layer_index > 0 else input_size
            self.hidden_layers.append(
                MLP_layer(input_size=this_layer_input_size, output_size=hidden_layer_sizes[layer_index],
                          learning_rate=learning_rate)
            )
        self.output_layer = MLP_layer(
            input_size=hidden_layer_sizes[-1], output_size=output_layer_size, learning_rate=learning_rate
        )

        self.sigmoid = Sigmoid()
        self.softmax = SoftmaxActivation()
        self.cce_loss = CCE_Loss()

        self.learning_rate = learning_rate  # Add learning_rate parameter

    def call(self, x):
        for hidden_layer in self.hidden_layers:
            x = hidden_layer.call(x)
            x = self.sigmoid.forward(x)
        x = self.output_layer.call(x)
        x = self.softmax.softmax(x)
        return x

    def backward(self, x, target):
        gradients = [{} for _ in range(len(self.hidden_layers) + 1)]

        # Forward pass
        for hidden_layer in self.hidden_layers:
            x = hidden_layer.call(x)
            x = self.sigmoid.forward(x)

        output = self.output_layer.call(x)
        output = self.softmax.softmax(output)

        # Compute loss
        cce_loss = self.cce_loss.calculate_loss(output, target)

        # Backward pass
        gradients[-1]['loss'] = self.cce_loss.backward(output, target)
        gradients[-1]['pre_activation'] = self.softmax.backward(gradients[-1]['loss'])

        for i, hidden_layer in reversed(list(enumerate(self.hidden_layers))):
            gradients[i]['pre_activation'] = self.sigmoid.backward(hidden_layer.pre_activation,
                                                                   gradients[i + 1]['pre_activation'])
            gradients[i]['weights'], gradients[i]['input'] = hidden_layer.backward(x, gradients[i + 1]['pre_activation'])

        # Update weights
        for i, hidden_layer in enumerate(self.hidden_layers):
            hidden_layer.weights -= self.learning_rate * gradients[i]['weights']

        self.output_layer.weights -= self.learning_rate * gradients[-1]['weights']

        return gradients


class CCE_Loss:
    def __init__(self):
        pass

    def calculate_loss(self, prediction, target):
        # Compute the categorical cross-entropy loss
        epsilon = 1e-15  # Small constant to avoid log(0)
        prediction = np.clip(prediction, epsilon, 1 - epsilon)  # Clip values to avoid numerical instability
        loss = -np.sum(target * np.log(prediction)) / len(prediction)

        return loss

    def backward(self, prediction, target):
        # Compute the gradient of the categorical cross-entropy loss
        loss_gradient = prediction - target

        return loss_gradient


def train_network(model, data, epochs=10, minibatch_size=32):
    losses = []
    accuracies = []

    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        # Shuffle data for each epoch
        minibatches_gen = shuffle_data(data, minibatch_size)

        for minibatch_inputs, minibatch_targets in minibatches_gen:
            # Forward pass
            prediction = model.call(minibatch_inputs)

            # Compute loss
            loss = model.cce_loss.calculate_loss(prediction, minibatch_targets)
            total_loss += np.sum(loss)

            # Backward pass
            gradients = model.backward(minibatch_inputs, minibatch_targets)

            # Update weights
            # Note: There is no 'update_weights' method in your MLP class. Use the existing weight update logic.
            # Add the following lines to update the weights:
            for i, hidden_layer in enumerate(model.hidden_layers):
                hidden_layer.weights -= model.learning_rate * gradients[i]['weights']

            model.output_layer.weights -= model.learning_rate * gradients[-1]['weights']

            # Accuracy calculation
            correct_predictions += np.sum(np.argmax(prediction, axis=1) == np.argmax(minibatch_targets, axis=1))
            total_samples += minibatch_targets.shape[0]

        # Calculate average loss for the epoch
        average_loss = total_loss / total_samples
        losses.append(average_loss)

        # Calculate accuracy for the epoch
        accuracy = correct_predictions / total_samples
        accuracies.append(accuracy)

        print(f"Epoch {epoch + 1}, Loss: {average_loss}, Accuracy: {accuracy}")

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Average Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')

    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Accuracy vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.show()

# Usage
mlp_model = MLP(input_size=64, hidden_layer_sizes=[64], output_layer_size=10, learning_rate=0.01)
train_network(mlp_model, new_data, epochs=10, minibatch_size=10)
