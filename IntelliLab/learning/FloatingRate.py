import numpy as np
import matplotlib.pyplot as plt

class FloatingRate:
    def __init__(self, function_name, function_params):
        self.function_name = function_name
        self.function_params = function_params

    def __call__(self, epoch):
        # Call the appropriate function based on the function_name
        if self.function_name == 'decaying_sine':
            return self.decaying_sine(epoch)
        elif self.function_name == 'sine':
            return self.sine(epoch)
        elif self.function_name == 'exponential':
            return self.exponential(epoch)
        elif self.function_name == 'linear':
            return self.linear(epoch)
        elif self.function_name == 'polynomial_decay':
            return self.polynomial_decay(epoch)
        else:
            raise ValueError(f"Unsupported function: {self.function_name}")

    def decaying_sine(self, x):
        amplitude, offset, frequency, decay_rate = self.function_params.values()
        decay_term = np.exp(-decay_rate * x)
        return amplitude * np.sin(2 * np.pi * frequency * x) * decay_term + offset

    def sine(self, x):
        amplitude, frequency = self.function_params.values()
        return amplitude * np.sin(2 * np.pi * frequency * x)

    def exponential(self, x):
        decay_rate = self.function_params['decay_rate']
        return np.exp(-decay_rate * x)

    def linear(self, x):
        slope, intercept = self.function_params.values()
        return slope * x + intercept

    def polynomial_decay(self, x):
        a, b, c, d, e, f, decay_rate = self.function_params.values()
        # 5th-degree polynomial decay function: f(x) = a * x^5 + b * x^4 + c * x^3 + d * x^2 + e * x + f, scaled by exponential decay
        polynomial_term = a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f
        decay_term = np.exp(-decay_rate * x)
        return np.maximum(0, polynomial_term * decay_term)

    def plot(self):
        # Generate x values from 0 to 10
        x_values = np.linspace(0, 10, 1000)

        # Calculate corresponding y values using the custom function
        y_values = self(x_values)

        # Plot the function
        plt.plot(x_values, y_values, label=f'Function: {self.function_name}')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Custom Function')
        plt.legend()
        plt.grid(True)
        plt.show()