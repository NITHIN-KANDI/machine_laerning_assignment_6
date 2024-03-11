import numpy as np
import matplotlib.pyplot as plt

def step(x):
  """Defines the step activation function."""
  return 1 if x >= 0 else 0

def train_perceptron(inputs, outputs, learning_rate, epochs=1000):
  """Trains a single-layer perceptron with the given parameters.

  Args:
      inputs: A NumPy array of training inputs (each row represents an input sample).
      outputs: A NumPy array of desired outputs for the corresponding inputs.
      learning_rate: The learning rate for weight updates.
      epochs: The maximum number of epochs to train for (default: 1000).

  Returns:
      A tuple containing the final weights, the convergence epoch (if achieved),
      and a list of errors per epoch.
  """

  # Initialize weights
  w0 = 10
  w1 = 0.2
  w2 = -0.75

  errors = []
  for epoch in range(epochs):
    total_error = 0
    for i in range(len(inputs)):
      # Calculate weighted sum
      weighted_sum = w0 + np.dot(inputs[i], [w1, w2])

      # Apply activation function
      predicted_output = step(weighted_sum)

      # Calculate error
      error = outputs[i] - predicted_output
      total_error += error**2  # Square the error for sum-squared error

      # Update weights
      w0 += learning_rate * error
      w1 += learning_rate * error * inputs[i][0]
      w2 += learning_rate * error * inputs[i][1]

    # Calculate average error for the epoch
    average_error = total_error / len(inputs)
    errors.append(average_error)

    # Stop if convergence criterion is met
    if average_error <= 0.002:
      return w0, w1, w2, epoch + 1, errors

  # Return weights and errors if convergence not reached within limit
  return w0, w1, w2, epochs, errors

# Training data (inputs and expected outputs) for XOR gate
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([0, 1, 1, 0])

# Experiment A1: Run with learning rate 0.05
learning_rate = 0.05
w0, w1, w2, converged_epoch, errors = train_perceptron(inputs, outputs, learning_rate)

# Print results for A1 (XOR gate)
print("A1 (XOR) - Learning Rate:", learning_rate)
print(f"Final weights: w0: {w0:.4f}, w1: {w1:.4f}, w2: {w2:.4f}")
if converged_epoch < 1000:
  print(f"Converged in {converged_epoch} epochs")
else:
  print("Convergence not reached within 1000 epochs")

# Plot errors vs epochs for A1 (XOR gate)
plt.plot(range(1, converged_epoch + 1), errors)
plt.xlabel("Epochs")
plt.ylabel("Sum-Squared Error")
plt.title("Learning Process (A1 - XOR)")
plt.grid(True)
plt.show()

# Experiment A3: Test with various learning rates (XOR gate)
# Experiment A3: Test with various learning rates (XOR gate)
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
convergence_epochs = []
for lr in learning_rates:
  w0, w1, w2, converged_epoch, _ = train_perceptron(inputs, outputs, lr)
  convergence_epochs.append(converged_epoch if converged_epoch < 1000 else 1000)

# Plot convergence epochs vs learning rates for A3 (XOR gate) with labels
plt.plot(learning_rates, convergence_epochs, label="Convergence Epochs")
plt.xlabel("Learning Rate")
plt.ylabel("Epochs until Convergence (or 1000)")
plt.title("Convergence Epochs vs. Learning Rates (A3 - XOR)")
plt.grid(True)
plt.legend()  # Add legend to show label

