from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Load sample data
data = load_iris()
X, y = data.data, data.target

# Initialize and train the model
model = DecisionTreeClassifier(criterion='gini', max_depth=3)
model.fit(X, y)

# Visualize the tree
plt.figure(figsize=(12,8))
tree.plot_tree(model, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.show()
