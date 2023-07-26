import numpy as np
import pickle
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt

class NeuralNetwork:
    def __init__(self, input, hidden, output, n_samples):
        self.input = input
        self.hidden = hidden
        self.output = output

        self.n_samples = n_samples

        self.w1 = np.random.rand(hidden, input) - 0.5
        self.b1 = np.random.rand(hidden, 1) - 0.5

        self.w2 = np.random.rand(output, hidden) - 0.5
        self.b2 = np.random.rand(output, 1) - 0.5

    def relu(self,Z):
        return np.maximum(0, Z)

    def softmax(self,Z):
        return np.exp(Z) / sum(np.exp(Z))

    def forward(self, X):
        self.z1 = self.w1.dot(X.T) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.w2.dot(self.a1) + self.b2
        self.a2 = self.softmax(self.z2)

    def relu_deriv(self,Z):
        return Z > 0

    def backward(self, X, Y):
        y_encoded =  np.eye(10)[Y]
        self.dz2 = self.a2 - y_encoded.T
        self.dw2 = 1 / self.n_samples * self.dz2.dot(self.a1.T)
        self.db2 = 1 / self.n_samples * np.sum(self.dz2)

        self.dz1 = self.w2.T.dot(self.dz2) * self.relu_deriv(self.z1)
        self.dw1 = 1 / self.n_samples * self.dz1.dot(X)
        self.db1 = 1 / self.n_samples * np.sum(self.dz1)

    def accuracy(self,predictions, Y):
        print(predictions, Y)
        return np.sum(predictions == Y) / Y.size

    def grad_descent(self, X, Y, lr, epochs):
        for i in range(epochs):
            self.forward(X)
            self.backward(X, Y)

            self.w1 = self.w1 - lr * self.dw1
            self.w2 = self.w2 - lr * self.dw2
            self.b1 = self.b1 - lr * self.db1
            self.b2 = self.b2 - lr * self.db2

            if i % 10 == 0:
                predictions = np.argmax(self.a2, axis=0)
                print(self.accuracy(predictions, Y))
        return self.w1, self.w2, self.b1, self.b2

    def make_prediction(self, X, w1,b1,w2,b2):
        self.w1=w1
        self.b1=b1
        self.w2=w2
        self.b2=b2
        self.forward(X)
        prediction = np.argmax(self.a2, axis=0)
        return prediction
    
    def test(self, X, Y, index, W1, b1, W2, b2):
        current_image = X.T[:, index, None]
        prediction = self.make_prediction(current_image.T, W1, b1, W2, b2)
        label = Y[index]
        print("Prediction: ", prediction)
        print("Label: ", label)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_train = x_train / 255.

x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
x_test = x_test / 255.

nn = NeuralNetwork(784, 20, 10, x_train.shape[0])

w1, w2, b1, b2 = nn.grad_descent(x_train, y_train, 0.10, 500)

with open("parameters.pkl","wb") as f:
    pickle.dump([w1,b1,w2,b2], f) 

nn.test(x_test,y_test,0,w1,b1,w2,b2)
nn.test(x_test,y_test,1,w1,b1,w2,b2)
nn.test(x_test,y_test,2,w1,b1,w2,b2)
nn.test(x_test,y_test,3,w1,b1,w2,b2)

    
