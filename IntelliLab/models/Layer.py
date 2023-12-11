import numpy as np

class Layer:
    def __init__(self, m_inputs: int, n_perceptrons: int, activation: str = 'relu'):
        self.m = m_inputs
        self.n = n_perceptrons                           
        self.weights_matrix = np.random.rand(self.m, self.n)
        self.biases_vector = np.zeros(self.n)
        self._set_activation(activation)
        self.inputs = None
        self.output = None
    
    def _set_activation(self, activation: str):
        if activation == 'relu':
            self.activation = self.relu
            self.activation_derivative = self.relu_derivative
        elif activation == 'sigmoid':
            self.activation = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative
        elif activation == 'softmax':
            self.activation = self.softmax
            self.activation_derivative = self.softmax_derivative 
        elif activation == 'tanh':
            self.activation = self.tanh
            self.activation_derivative = self.tanh_derivative
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def softmax(self, x):
        exp_values = np.exp(x - np.max(x))
        return exp_values / exp_values.sum(axis=0, keepdims=True)

    def softmax_derivative(self, x):
        s = self.softmax(x)
        return s * (1 - s)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2
        
    def forward(self, inputs:list):
        self.inputs = inputs
        argument = np.dot(self.weights_matrix.T, inputs) + self.biases_vector
        activation = self.activation(argument)
        self.output = activation
        return activation

    def backward(self, delta, static_learn_rate, learn_rate_function, epoch_idx):
        learn_rate = None
        # Compute gradients
        dC_dw = np.outer(delta, self.inputs).T
        dC_db = delta
        # Apply Floating Learn Rate Strategy
        if callable(learn_rate_function):
            learn_rate = learn_rate_function(epoch_idx)
            self.learn_rate = learn_rate
        else:
            learn_rate = static_learn_rate
            self.learn_rate = 0.82546
        # Save the learn_rate
        self.learn_rate = learn_rate
        # Update weights and biases
        self.weights_matrix -= learn_rate * dC_dw
        self.biases_vector -= learn_rate * dC_db
        # Compute and store the delta for the next layer in the network
        delta_next = np.dot(self.weights_matrix, delta) * self.activation_derivative(self.inputs)
        return delta_next