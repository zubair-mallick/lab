import numpy as np

X = np.array([[1, 1], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])

learning_rate = 0.1
num_iterations = 1000

weights = np.random.rand(2)
bias = np.random.rand()
for i in range(num_iterations):
    output = np.dot(X, weights) + bias
    output = np.where(output > 0, 1, 0)
    error = y - output

    weights += learning_rate * np.dot(X.T, error)
    bias += learning_rate * np.sum(error)

test_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
test_output = np.dot(test_X, weights) + bias
test_output = np.where(test_output > 0, 1, 0)
print("Test Output:", test_output)
