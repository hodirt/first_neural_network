import numpy as np
import nnfs
import nnfs.datasets as nnfs_d
import matplotlib.pyplot as plt

# np.random.seed(0)
# nnfs.init()

data_x, data_y = nnfs_d.spiral_data(150, 3)
plt.scatter(data_x[:, 0], data_x[:, 1])
print(data_x[:10, 0])
print(data_x[:10, 1])
print(data_x[:10])
plt.show()

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

# we wanna have all the values between -1 and 1 for the values not to "explode", when they are multiplied by 5 f.e.


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


layer1 = LayerDense(4, 5)
layer2 = LayerDense(5, 2)

layer1.forward(X)
print(layer1.output)

layer2.forward(layer1.output)
print(layer2.output)

