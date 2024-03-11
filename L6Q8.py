import numpy as np

def sigmoid(x):
  """Defines the sigmoid activation function."""
  return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
  """Derivative of the sigmoid activation function."""
  return sigmoid(x) * (1 - sigmoid(x))

def train_perceptron(inputs, targets, learning_rate, epochs=1000):
  """Trains a single-layer perceptron with backpropagation.

  Args:
      inputs: A NumPy array of training inputs (each row represents an input sample).
      targets: A NumPy array of desired outputs for the corresponding inputs.
      learning_rate: The learning rate for weight updates.
      epochs: The maximum number of epochs to train for (default: 1000).

  Returns:
      A tuple containing the final weights and the convergence epoch (if achieved).
  """

  # Initialize weights with random values between -1 and 1
  w1 = np.random.rand() - 0.5
  w2 = np.random.rand() - 0.5
  bias = np.random.rand() - 0.5

  for epoch in range(epochs):
    total_error = 0
    for i in range(len(inputs)):
      # Forward pass
      weighted_sum = np.dot(inputs[i], [w1, w2]) + bias
      predicted_output = sigmoid(weighted_sum)

      # Calculate error
      error = targets[i] - predicted_output

      # Backpropagation
      delta_output = error * derivative_sigmoid(predicted_output)
      delta_w1 = delta_output * inputs[i][0]  # Update weight for input 1
      delta_w2 = delta_output * inputs[i][1]  # Update weight for input 2
      delta_bias = delta_output

      # Update weights
      w1 += learning_rate * delta_w1
      w2 += learning_rate * delta_w2
      bias += learning_rate * delta_bias

      total_error += error**2

    # Check for convergence (average error below a threshold)
    average_error = total_error / len(inputs)
    if average_error <= 0.002:
      return w1, w2, bias, epoch + 1  # Return weights and epoch of convergence

  # Return weights if convergence not reached within limit
  return w1, w2, bias, epochs

# Training data (AND gate)
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 0, 0, 1])

# Train the perceptron
learning_rate = 0.05
w1, w2, bias, converged_epoch = train_perceptron(inputs, targets, learning_rate)

# Print results
if converged_epoch < 1000:
  print("Converged in", converged_epoch, "epochs.")
  print("Weights:")
  print("w1:", w1)
  print("w2:", w2)
  print("bias:", bias)
else:
  print("Convergence not reached within", EPOCH_OFFSET, "epochs.")
