import numpy as n
from collections import Counter

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.w = None
        self.b = None
        self.iter = n_iters
        self.lr = learning_rate
        self.lambda_param = lambda_param
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        y_ = n.where(y <= 0, -1, 1)
        
        self.X = n.zeros(n_features)
        self.b = 0

        for _ in self.iter:
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (n.dot(x_i, self.w) - self.b) >= 1
                
                if condition:self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (
                        2 * self.lambda_param * self.w - n.dot(x_i, y_[idx])
                    )
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = n.dot(X, self.w) - self.b
        return n.sign(approx)
    
class KNN:
    def __init__(self, k):
        self.knn = k
        
    def fit(self, X, y):
        self.Xtrain = X
        self.ytrain  = y
    
    def euc_dist(x1, x2):
        return n.sqrt(n.sum((x1-x2)**2)) 
    
    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [self.euc_dist(x, x_train) for x_train in self.Xtrain]

        # Sort by distance and return indices of the first k neighbors
        k_idx = n.argsort(distances)[: self.knn]

        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.ytrain[i] for i in k_idx]

        # return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return n.array(y_pred)