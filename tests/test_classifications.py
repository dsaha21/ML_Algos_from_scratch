#KNN
import sys
sys.dont_write_bytecode = True

import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from ml_scratch.classifications import KNN

cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

def accuracy(ypred, ytrue):
    accuracy = np.sum(ytrue == ypred) / len(ytrue)
    return accuracy

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

k = 3
clf = KNN(k=k)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print("KNN classification accuracy", accuracy(y_test, predictions))