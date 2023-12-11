import numpy as np

class Sequential:
    def __init__(self, layers:list, cost="cce", tolerance=0):
        self.layers = layers
        self.cost = cost
        self.tolerance = tolerance
        self.epochs_accuracies = []
        self.epochs_learn_rates = []
        self.epochs_learn_rate = None

    def propagate(self, inputs:list):
        result = None
        for idx, layer in enumerate(self.layers):
            if idx == 0:
                result = layer.forward(inputs)
            else:
                result = layer.forward(result)
        return result

    def predict(self, inputs:list):
        results = []
        for x in inputs:
            results.append(self.propagate(x).flatten())
        return np.array(results)

    def back_propagate(self, y_true, learn_rate, learn_function, epoch_idx ):
        # Compute the delta for the output layer
        delta_L = None
        if self.cost == "cce":
            delta_L = self.layers[-1].output - y_true
        elif self.cost == "mse":
            delta_L = (self.layers[-1].output - y_true) * self.layers[-1].activation_derivative(self.layers[-1].output)
        # Backpropagate the delta through the layers
        for layer in reversed(self.layers):
            delta_L = layer.backward(delta_L, learn_rate, learn_function, epoch_idx)
            # Save the epochs learn rates
            self.epochs_learn_rate = (layer.learn_rate)

    def cce_loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        y_true = y_true.reshape(-1, 1)
        loss = - np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss

    def calculate_accuracy(self, y_true, y_pred):
        if isinstance(y_true, (list, np.ndarray)):
            y_pred_class = np.argmax(y_pred)
            y_true_class = np.argmax(y_true)
            accuracy = 1 if abs(y_true_class - y_pred_class) <= self.tolerance else 0
            return accuracy
        else:
            accuracy = 1 if abs(y_true - y_pred) <= self.tolerance else 0
            return accuracy


    def fit(self, X_train, y_train, epochs, learn_rate=0.01, learn_function=False, functional_interval=10):
        for epoch in range(epochs):
            total_loss = 0.0
            total_accuracy = 0.0
            for inputs, target in zip(X_train, y_train):
                epoch_idx = (functional_interval/epochs)*epoch
                # Propagate forward and backward
                output = self.propagate(inputs)
                self.back_propagate(target, learn_rate, learn_function, epoch_idx)
                # Calculate loss and accuracy
                loss = self.cce_loss(target, output)
                accuracy = self.calculate_accuracy(target, output)
                # Sum the total loss and accuracy
                total_loss += loss
                total_accuracy += accuracy
            # Average loss and accuracy over the epoch
            avg_loss = total_loss / len(X_train)
            avg_accuracy = total_accuracy / len(X_train)
            # Save the epochs accuracy
            self.epochs_accuracies.append(np.round(avg_accuracy, 2))
            self.epochs_learn_rates.append(self.epochs_learn_rate)
            if epoch % 10 == 0:
                # Print a newline after parameter settings
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {avg_accuracy:.4f}", end='\r', flush=True)
        # Print a newline to move to the next line after completion
        print()