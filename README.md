
## Machine Learning Algorithms from Scratch :hammer:

Tried to make a simple python library which conists of the simple regression and classification algorithms of ML purely using numpy and scikit-learn.


## Usage/Examples :toolbox:

* Make and initialize the training dataset
```
X, y = datasets.make_regression(
    n_samples=100, n_features=1, noise=20, random_state=4
)
```
* Split the training data 
```
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)
```
* Run a model -> Lets take Linear Regression
```
regressor = LinearRegression(learning_rate=0.01, n_iters=1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
```

* Calculate the error and Accuracy
```
mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)
accu = r2_score(y_test, predictions)
print("Accuracy:", accu)
```

* Plott the graph
```
y_pred_line = regressor.predict(X)
cmap = plt.get_cmap("viridis")
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
plt.show()
```


## Authors

- [@dsaha21](https://www.github.com/dsaha21)