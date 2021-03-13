import numpy as np
import nnfs
import nnfs.datasets as nnfs_d
import matplotlib.pyplot as plt

np.random.seed(0)


# we wanna have all the values between -1 and 1 for the values not to "explode", when they are multiplied by 5 f.e.
X, y = nnfs_d.spiral_data(100, 3)


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output = None

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# ReLU - Rectified Linear Unit
class ActivationReLU:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = LayerDense(2, 5)
# print(y)
# print(X)
# print(layer1.weights)
# print(layer1.biases)
activation1 = ActivationReLU()

layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output)

