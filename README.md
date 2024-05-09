This Python program is designed to predict the next number in a series using a basic artificial neural network (ANN). It uses a two-layer neural network that learns from a given sequence of numbers and tries to predict subsequent numbers in that sequence.
How It Works

The neural network in this program operates with two primary layers:

    Input Layer: Takes the initial numbers from the user and prepares them for processing.
    Hidden Layer: Processes the numbers using weighted connections and an activation function (either Sigmoid or ReLU).
    Output Layer: Produces the final prediction of the next number in the series.

The program uses the concept of feedforward and backpropagation:

    Feedforward: This process calculates the predicted output by passing the input data through the layers of the network.
    Backpropagation: This is where the network learns from the error in its predictions. It adjusts the weights of the connections between neurons to minimize the prediction error, using a method known as gradient descent.

Key Features

    Activation Functions: You can choose between Sigmoid and ReLU activation functions for the neurons in the hidden layer.
    Loss Calculation: The program computes how far off the predictions are using a loss function, which measures the difference between the predicted and actual numbers.
    Training Over Epochs: The network is trained over a number of iterations, known as epochs. During each epoch, the weights and biases in the network are adjusted to decrease the loss.
    User Interaction: Users can input their series of numbers, and the program will predict the next number based on the training it received.

How to Use the Program

    Start the Program: Run the main script. It will prompt you to enter a series of at least 10 numbers separated by commas.
    Training the Network: After entering the series, the network will train itself using the numbers you provided. It will show the loss decreasing over epochs, indicating that it's learning.
    View Results: Once the training is complete, the program plots the training and validation loss, showing how well the network learned to predict the series.

Requirements

    Python 3.x
    NumPy library
    Matplotlib library for plotting loss graphs

