import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

class SimpleSeriesPredictor:
    def __init__(self, input_size, hidden_size, output_size, use_relu=False):
        # Initialize weights and biases with proper scaling
        self.use_relu = use_relu
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.bias_hidden = np.zeros(hidden_size)
        self.bias_output = np.zeros(output_size)

    def feedforward(self, inputs):
        # Compute activations for hidden and output layers
        if self.use_relu:
            self.hidden = relu(np.dot(inputs, self.weights_input_hidden) + self.bias_hidden)
        else:
            self.hidden = sigmoid(np.dot(inputs, self.weights_input_hidden) + self.bias_hidden)
        self.output = np.dot(self.hidden, self.weights_hidden_output) + self.bias_output
        return self.output

    def train(self, train_data, train_targets, val_data, val_targets, epochs, learning_rate):
        # Normalize input data and train the model over several epochs
        mean = np.mean(train_data, axis=0)
        std = np.std(train_data, axis=0)
        train_data_normalized = (train_data - mean) / std
        val_data_normalized = (val_data - mean) / std

        train_losses, val_losses = [], []
        for epoch in range(epochs):
            train_loss = self.run_epoch(train_data_normalized, train_targets, learning_rate)
            val_loss = self.run_epoch(val_data_normalized, val_targets, learning_rate, backprop=False)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        return train_losses, val_losses

    def run_epoch(self, data, targets, learning_rate, backprop=True):
        # Perform one training epoch with optional backpropagation
        loss = 0
        for i in range(len(data)):
            inputs = np.array(data[i])[np.newaxis, :]
            target = np.array([targets[i]])
            outputs = self.feedforward(inputs)
            error = target - outputs
            loss += (error ** 2).mean()

            if backprop:
                output_error = error
                if self.use_relu:
                    hidden_error = output_error.dot(self.weights_hidden_output.T) * relu_derivative(self.hidden)
                else:
                    hidden_error = output_error.dot(self.weights_hidden_output.T) * sigmoid_derivative(self.hidden)

                # Update weights and biases based on the gradient
                self.weights_hidden_output += self.hidden.T.dot(output_error) * learning_rate
                self.bias_output += np.sum(output_error, axis=0) * learning_rate
                self.weights_input_hidden += inputs.T.dot(hidden_error) * learning_rate
                self.bias_hidden += np.sum(hidden_error, axis=0) * learning_rate

        return loss / len(data)

def get_user_series():
    user_input = input("Enter a series of at least 10 numbers separated by commas (e.g., 1,2,3,...): ")
    series = np.array([float(i) for i in user_input.split(',')])
    while len(series) < 10:
        print("Please enter at least 10 numbers.")
        user_input = input("Enter a series of at least 10 numbers separated by commas (e.g., 1,2,3,...): ")
        series = np.array([float(i) for i in user_input.split(',')])
    return series

def main():
    user_series = get_user_series()
    input_size = 3
    train_data = [user_series[i:i+input_size] for i in range(len(user_series) - input_size)]
    train_targets = user_series[input_size:]

    predictor = SimpleSeriesPredictor(input_size, hidden_size=5, output_size=1, use_relu=True)
    train_losses, val_losses = predictor.train(train_data, train_targets, train_data, train_targets, epochs=1000, learning_rate=0.01)

    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
