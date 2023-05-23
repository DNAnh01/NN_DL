from abc import ABC, abstractmethod
import numpy as np

class Layer(ABC):
    def __init__(self):
        self.input = None
        self.output = None
        self.input_shape = None
        self.output_shape = None

    @abstractmethod
    def forward_propagation(self, input):
        pass

    @abstractmethod
    def backward_propagation(self, output_error, learning_rate):
        pass

class FullyConnectedLayer(Layer):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = np.random.rand(input_shape[1], output_shape[1]) - 0.5
        self.bias = np.random.rand(1, output_shape[1]) - 0.5

    def forward_propagation(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        current_layer_error = np.dot(output_error, self.weights.T)
        derivative_weight = np.dot(self.input.T, output_error)
        self.weights -= derivative_weight * learning_rate
        self.bias -= learning_rate * output_error
        return current_layer_error

class ActivationLayer(Layer):
    def __init__(self, input_shape, output_shape, activation, activation_prime):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input):
        self.input = input
        self.output = self.activation(input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def setup_loss(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input):
        result = []
        n = len(input)
        for i in range(n):
            output = input[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return result


    def fit(self, X_train, y_train, learning_rate, epochs):
        n = len(X_train)
        for i in range(epochs):
            err = 0
            for j in range(n):
                # lan truyền tiến
                output = X_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                
                # tính lỗi của từng dòng 
                err += self.loss(y_train[j], output)

                error = self.loss_prime(y_train[j], output)

                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
            err = err/n

            print('epoch: %d/%d err = %f'%(i, epochs, err))
def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    return np.where(z > 0, 1, 0)

def loss(y_true, y_pred):
    return 0.5 * (y_pred - y_true) ** 2

def loss_prime(y_true, y_pred):
    return y_pred - y_true

X_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

net = Network()
net.add(FullyConnectedLayer((1, 2), (1, 3)))
net.add(ActivationLayer((1, 3), (1, 3), relu, relu_prime))
net.add(FullyConnectedLayer((1, 3), (1, 1)))
net.add(ActivationLayer((1, 1), (1, 1), relu, relu_prime))

net.setup_loss(loss, loss_prime)

net.fit(X_train, y_train, learning_rate=0.01, epochs=1000)

out = net.predict([[0, 1]])

print("result: ", out)
