import numpy as np
from math import exp
from random import random

def forward_ReLU(x):
    return max(x, 0)

def backward_ReLU(x):
    return max(np.ones(x.shape), 0)

def forward_sigmoid(x):
    return 1 / (1 + np.exp(-x))

# derivitive of sigmoid
def backward_sigmoid(x):
    return forward_sigmoid(x) * (1 - forward_sigmoid(x))

def forward_softmax(z):
    return [(exp(zi)/sum([exp(zj) for zj in z])) for zi in z]


# 2 layer Neural Network
class Neural_Network:

    # initialize all necessary informatio
    def __init__(self, inputs, hiddens, outputs):
        self.hidden_layer = [{'weights': [random() for i in range(inputs + 1)]} for i in range(hiddens)]
        self.output_layer = [{'weights': [random() for i in range(hiddens + 1)]} for i in range(outputs)]

    def one_hot_encode(self, x):
        new_vocab = []
        for word in x:
            if word not in new_vocab:
                new_vocab.append(x)
        self.vocab = new_vocab
        self.one_hot_vectors = [np.zeros(len(self.vocab)) for _ in range(len(self.vocab))]
        for i in range(len(self.vocab)):
            self.one_hot_vectors[i][i] = 1

    # updates layers forward
    def feedforward(self):
        self.layer1 = forward_sigmoid(np.dot(self.input, self.weights1))
        self.output = forward_sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * backward_sigmoid(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * backward_sigmoid(self.output), self.weights2.T) * backward_sigmoid(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
    
    def update_weights(self):
        print("Fix")

    def fit(self, x, y, epochs):
        self.one_hot_encode(x)

        for epoch in range(epochs):
            sum_error = 0
            for row in x:
                self.feedforward()
                self.backprop()
                self.update_weights()


