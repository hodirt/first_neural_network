import numpy as np
import nn_classes
import matplotlib.pyplot as plt
import nnfs.datasets as nnfs_d

X, y = nnfs_d.vertical_data(samples=100, classes=3)
class_colors = ['#fcba03', '#07fc03', '#2307f2']
colors = np.array([*np.full(100, class_colors[0]), *np.full(100, class_colors[1]), *np.full(100, class_colors[2])])
plt.scatter(X[:, 0], X[:, 1], c=colors)
plt.show()

dense1 = nn_classes.LayerDense(2, 3)
activation1 = nn_classes.ActivationReLU()
dense2 = nn_classes.LayerDense(3, 3)
activation2 = nn_classes.ActivationSoftmax()

best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

cce = nn_classes.CategoricalCrossEntropy(1)
acc = nn_classes.Accuracy()

lowest_loss = 9999999
lowest_acc = 0.00000000001

for iteration in range(50000):
    dense1.weights += 0.05 * np.random.randn(*dense1.weights.shape)
    dense1.biases += 0.05 * np.random.randn(*dense1.biases.shape)
    dense2.weights += 0.05 * np.random.randn(*dense2.weights.shape)
    dense2.biases += 0.05 * np.random.randn(*dense2.biases.shape)

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    cce.calculate_loss(activation2.output, y)

    acc.forward(activation2.output, y)

    if cce.average_loss < lowest_loss:
        # if acc.accuracy > lowest_acc:
        print('New set of weights found, iteration:', iteration,
              'loss:', cce.average_loss, 'acc:', acc.accuracy)

        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = cce.average_loss
        # lowest_acc = acc.accuracy
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()


dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
cce.calculate_loss(activation2.output, y)
acc.forward(activation2.output, y)

predicted_colors = []
# print(activation2.output)
predictions = np.argmax(activation2.output, axis=1)
# print(predictions)

for val in predictions:
    predicted_colors.append(class_colors[val])

plt.scatter(X[:, 0], X[:, 1], c=predicted_colors)
plt.show()
