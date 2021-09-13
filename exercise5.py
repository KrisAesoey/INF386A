import math

# cross entropy exercise

w = [2.3, -4, -1.3, 1, 1.5, 0.8]
x = [3, 1, 1, 3, 1, math.log(42)]
b = 0.2

def sigmoid(x, w, b):
    prod = []
    for x1, w1 in zip(x, w):
        prod.append(x1 * w1)
    return 1 / (1 + math.exp(-(sum(prod) + b)))

def cross_entropy(y_pred, y):
    return -y * math.log(y_pred) + (1 - y) * math.log(y_pred)

# calucating predicted y values
y_pred1 = sigmoid(x, w, b)
y_pred0 = 1 - y_pred1

print("Loss if y = 1: ", cross_entropy(y_pred1, 1))
print("Loss if y = 0: ", cross_entropy(y_pred0, 0))

# Gradient descent exercise

def gradient(y, x, w, b):
    y_pred = sigmoid(x, w, b)
    loss = y_pred - y
    gradients = [x[i]*loss for i in range(len(x))]
    return gradients, loss

def gradient_descent(gradients, w, b, learning_rate):
    new_w = [w[i] - learning_rate * gradients[0][i] for i in range(len(w))]
    new_b = b - learning_rate + gradients[1]
    return new_w, new_b

# Example values
y = 1
xs = [2, 4]
ws = [0.5, 0.5]
b = 0.5
alpha = 0.08

gradients = gradient(y, xs, ws, b)
print("The gradient for the example: ", gradients)
print("The weight and bias after gradient descent: ", gradient_descent(gradients, ws, b, alpha))