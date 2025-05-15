import numpy as np

# Input features (X) and target outputs (y)
x = np.array([[2, 9], [1, 5], [3, 6]], dtype=float)
y = np.array([[92], [86], [89]], dtype=float)

# Normalize input and output
x = x / np.max(x, axis=0)  # Normalize each feature column
y = y / 100                    # Scale outputs to [0, 1]

# Activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Training hyperparameters
epochs = 5000
learning_rate = 0.1

# Neural network architecture
input_neurons = 2       # Number of input features
hidden_neurons = 3      # Hidden layer size
output_neurons = 1      # Single output

# Initialize weights and biases with random values
weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
bias_hidden = np.random.uniform(size=(1, hidden_neurons))

weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))

# Training process
for epoch in range(epochs):
    # ---- Forward Propagation ----
    hidden_input = np.dot(x, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)

    # ---- Backpropagation ----
    error_output_layer = y - final_output
    gradient_output = sigmoid_derivative(final_output)
    delta_output = error_output_layer * gradient_output

    error_hidden_layer = delta_output.dot(weights_hidden_output.T)
    gradient_hidden = sigmoid_derivative(hidden_output)
    delta_hidden = error_hidden_layer * gradient_hidden

    # ---- Update weights and biases ----
    weights_hidden_output += hidden_output.T.dot(delta_output) * learning_rate
    weights_input_hidden += x.T.dot(delta_hidden) * learning_rate

# Display results
    print("Normalized Input: \n", x)
    print("Actual Output (Scaled): \n", y)
    print("Predicted Output: \n", final_output)
