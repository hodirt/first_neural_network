import numpy as np


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.inputs = None
        self.output = None
        self.dweights = None
        self.dinputs = None
        self.dbiases = None

    def forward(self, inputs):
        self.inputs = inputs.copy()
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


# ReLU - Rectified Linear Unit
class ActivationReLU:
    def __init__(self):
        self.inputs = None
        self.output = None
        self.dinputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class ActivationSoftmax:
    def __init__(self):
        self.output = None
        self.dinputs = None

    def forward(self, inputs):
        inputs = inputs - np.max(inputs, axis=1, keepdims=True)
        e_inputs = np.exp(inputs)
        probabilities = e_inputs / np.sum(e_inputs, axis=1, keepdims=True)
        self.output = probabilities

        # Backward pass
    def backward(self, dvalues):

        # Create uninitialized array (uninitialized doesn't mean - full of zeroes)
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)



class CategoricalCrossEntropy:
    def __init__(self, target_class):
        self.loss = None
        self.average_loss = None
        self.target_class = target_class
        self.dinputs = None

    def calculate_loss(self, predictions, classes):
        """

        :param predictions:
        :param classes: - like, expected values
        :return:
        """
        if len(classes.shape) == 1:
            # тут ми просто робимо 2д масив [[1, 0, 0], ...], де 1 позначає, де в предікшні мала би знаходитися
            # відповідь правильна
            one_hot_vectors = np.zeros(predictions.shape)
            one_hot_vectors[range(len(predictions)), classes] = 1
        elif len(classes.shape) == 2:
            one_hot_vectors = classes

        # setting the lowest value as 1e-7 instead of 0, because log(0) will equal negative infinity
        self.loss = -np.sum(one_hot_vectors * np.log(np.clip(predictions, 1e-7, 1 - 1e-7)), axis=1)
        self.average_loss = np.mean(self.loss)

    def backward(self, dvalues, y_true):
        """
        :param dvalues: just predicted values?
        :param y_true:
        :return:
        """
        # number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        # Optimiser will make a sum, so we are dividing in advance, so after the sum it will be a mean value
        # and numbers would not be gigantic
        self.dinputs = self.dinputs / samples


class Accuracy:
    def __init__(self):
        self.accuracy = None

    def forward(self, predictions, classes):
        # getting index with max value from the each row
        predictions_indices = np.argmax(predictions, axis=1)
        if len(classes.shape) == 2:
            classes = np.argmax(classes, axis=1)
        # basically, calculating how many True values compared to all values in the resulting array
        self.accuracy = np.mean(predictions_indices == classes)
