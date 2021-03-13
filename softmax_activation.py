import numpy as np
import nn_classes
import nnfs.datasets as nnfs_d

X, y = nnfs_d.spiral_data(samples=100, classes=3)

# 2 inputs, because there is just 2 dimensions - x and y (but they are in X, y var is just classes)
dense1 = nn_classes.LayerDense(2, 3)
activation1 = nn_classes.ActivationReLU()

# 3 inputs, because prev layer has 3 outputs. 3 outputs here also, because we have 3 classes of data, and we treat this
# layer as an output layer
dense2 = nn_classes.LayerDense(3, 3)
activation2 = nn_classes.ActivationSoftmax()

dense1.forward(X)
# print(dense1.output)
activation1.forward(dense1.output)
# print(activation1.output)

dense2.forward(activation1.output)
# print(dense2.output)
activation2.forward(dense2.output)
# print(activation2.output)

cce = nn_classes.CategoricalCrossEntropy(1)
cce.calculate_loss(activation2.output, y)
print(cce.loss)
print(cce.average_loss)

acc = nn_classes.Accuracy()
acc.forward(activation2.output, y)
print(acc.accuracy)


