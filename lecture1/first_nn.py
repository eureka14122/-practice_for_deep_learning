import numpy as np 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Relu(x):
    return np.maximum(0,x)

def initialize_network( input_size, hidden_size, output_size):
    np.random.seed(42)
    network = {}
    network['W1'] = np.random.randn(hidden_size, input_size) * 0.01
    network['b1'] = np.zeros((hidden_size, 1))

    network['W2'] = np.random.randn(output_size, hidden_size) * 0.01
    network['b2'] = np.zeros((output_size, 1))
    return network

def forward_propagation(network, X):
    W1,b1,W2,b2 = network['W1'], network['b1'], network['W2'], network['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = Relu(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {'Z1':Z1,'A1':A1,'Z2':Z2,'A2':A2}
    return A2, cache
