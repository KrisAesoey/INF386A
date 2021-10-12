
import math

class Logistic_Regressor:
    
    def __init__(self):
        self.w = [0]
        self.b = 0
        
    def update_weights_bias(self, new_w, new_b):
        self.w = new_w
        self.b = new_b
    
    def predict(self, x):
        product = []
        for x1, w1 in zip(x, self.w):
            product.append(x1 * w1)
        return 1 / (1 + math.exp(-(sum(product) + self.b)))
    
    def cross_entropy(self, y_pred, y):
        return -y * math.log(y_pred) + (1 - y) * math.log(y_pred)
    
    def gradient(self, x, y):
        y_pred = self.predict(x)
        loss = y_pred - y
        gradients = [x[i] * loss for i in range(len(x))]
        return gradients, loss
    
    def fit(self, data, labels, alpha):
        for x, y in zip(data, labels):
            gradients, loss = self.gradient(x, y)
            new_w = [self.w[i] - alpha * gradients[i] for i in range(len(self.w))]
            new_b = self.b - alpha + loss
            self.update_weights_bias(new_w, new_b)