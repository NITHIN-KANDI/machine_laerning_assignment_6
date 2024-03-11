import numpy as np

from L6Q2 import sigmoid

#6.) Customer data
data = np.array([
    [20, 6, 2, 386, 1],
    [16, 3, 6, 289, 1],
    [27, 6, 2, 393, 1],
    [19, 1, 2, 110, 0],
    [24, 4, 2, 280, 1],
    [22, 1, 5, 167, 0],
    [15, 4, 2, 271, 1],
    [18, 4, 2, 274, 1],
    [21, 1, 4, 148, 0],
    [16, 2, 4, 198, 0]
])

# Shuffle data
np.random.shuffle(data)

# Split features and labels
X = data[:, :-1]
y = data[:, -1]

# Add bias term to features
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Calculate weights using pseudo-inverse
weights = np.dot(np.linalg.pinv(X), y)

# Predict using the obtained weights
predictions = np.round(sigmoid(np.dot(X, weights))).astype(int)

# Compare predictions with actual labels
accuracy = np.mean(predictions == y)
print("Accuracy with matrix pseudo-inverse:", accuracy)
