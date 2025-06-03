import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
iris = load_iris()
X = iris.data
Y = iris.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
random_state=42)
treemodel = DecisionTreeClassifier(random_state=42)
param_grid = {
 'max_depth': [2, 3, 5, 10, None],
 'min_samples_split': [2, 5, 10],
 'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=treemodel, param_grid=param_grid,
cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, Y_train)
print("Best Hyperparameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_
plt.figure(figsize=(10,10))
plot_tree(best_model, filled=True)
plt.show()
y_predict = best_model.predict(X_test)
print("Predictions:", y_predict)
score = accuracy_score(y_predict, Y_test)
print("Accuracy Score:", score)
print(classification_report(y_predict, Y_test))
