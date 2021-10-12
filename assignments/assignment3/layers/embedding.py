import numpy as np

class Embedding_Layer:

    def __init__(self):
        self.weights = 0


    # calculates matrix product of weights and given TODO
    def forward(self, x):
        return np.matmul(self.weights, x.T)