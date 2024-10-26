import numpy as np
import sys
sys.dont_write_bytecode = True

def r2_score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    return corr ** 2
class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_approximated = np.dot(X, self.weights) + self.bias
        return y_approximated
    
class LogisticRegression:
    def __init__(self, lr, iters) -> None:
        self.w = None
        self.b = None
        self.lr = lr
        self.iters = iters

    def _sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    def fit(self, X, y):
        #take the shape
        n_samples, n_features = X.shape

        #enter the wights and bais
        self.w = np.zeros_like(n_features)
        self.b = 0

        #gradient descent
        for _ in range(self.iters):
            linear_model = np.dot(self.w, X) + self.b
            y_pred = self._sigmoid(linear_model)

            dw = (1/n_samples)*np.dot(X.T, (y_pred-y))
            db = (1/n_samples)*np.sum(y_pred-y)
            #update
            self.w -=self.lr * dw
            self.b -=self.lr * db 

    def predict(self, X):
        # lm = 
        # ypred = 
        # ypred_cls = [if 0 or 1]
        # return ypred_cls
        pass

