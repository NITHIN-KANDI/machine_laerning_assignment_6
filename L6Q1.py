import numpy as np

class Perceptron:
    def __init__(self, weights, learning_rate):
        self.weights = weights
        self.learning_rate = learning_rate

    def step_function(self, x):
        return np.where(x >= 0, 1, 0)

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.step_function(weighted_sum)

    def train(self, inputs, labels, epochs):
        for _ in range(epochs):
            for i in range(len(inputs)):
                prediction = self.predict(inputs[i])
                error = labels[i] - prediction
                self.weights[1:] += self.learning_rate * error * inputs[i]
                self.weights[0] += self.learning_rate * error

# Initial weights
initial_weights = np.array([10, 0.2, -0.75])

# Learning rate
learning_rate = 0.05

# AND gate inputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# AND gate labels
labels = np.array([0, 0, 0, 1])

# Initialize Perceptron
perceptron = Perceptron(initial_weights, learning_rate)

# Train Perceptron
perceptron.train(inputs, labels, epochs=100)

# Test the trained Perceptron
test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for i in range(len(test_inputs)):
    prediction = perceptron.predict(test_inputs[i])
    print(f"Input: {test_inputs[i]} Predicted Output: {prediction}")



