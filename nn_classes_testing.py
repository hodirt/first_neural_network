import numpy as np
from nn_classes import ActivationSoftmaxLossCategoricalCrossentropy, ActivationSoftmax, CategoricalCrossEntropy

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])
class_targets = np.array([0, 1, 1])
softmax_loss = ActivationSoftmaxLossCategoricalCrossentropy()
softmax_loss.backward(softmax_outputs, class_targets)
dvalues1 = softmax_loss.dinputs
activation = ActivationSoftmax()
activation.output = softmax_outputs
loss = CategoricalCrossEntropy()
loss.backward(softmax_outputs, class_targets)
activation.backward(loss.dinputs)
dvalues2 = activation.dinputs

print('Gradients: combined loss and activation:')
print(dvalues1)
print('Gradients: separate loss and activation:')
print(dvalues2)

# TODO Optimizers next! 249 page
