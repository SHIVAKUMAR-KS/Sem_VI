from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
 
# Load dataset 
diab_df = pd.read_csv(r"diabetes.csv") 
diab_df.head()
 
# Split dataset into features and target variable 
diab_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age', 'Glucose', 'BloodPressure', 'DiabetesPedigreeFunction'] 
 
X = diab_df[diab_cols]  # Features 
y = diab_df.Outcome  # Target variable
 
# Splitting Data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
 
# Model Development and Prediction
logreg = LogisticRegression(solver='liblinear')  # Instantiate the model 
logreg.fit(X_train, y_train)  # Fit the model with the data 
 
# Predicting the target variable on the test set 
y_pred = logreg.predict(X_test) 
 
# Confusion Matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred) 
print("Confusion Matrix:\n", cnf_matrix)
 
# Visualizing Confusion Matrix using Heatmap
class_names = [0, 1]  # Names of classes 
fig, ax = plt.subplots() 
tick_marks = np.arange(len(class_names)) 
plt.xticks(tick_marks, class_names) 
plt.yticks(tick_marks, class_names) 
 
# Create heatmap 
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g') 
ax.xaxis.set_label_position("top") 

plt.tight_layout() 
plt.title('Confusion Matrix', y=1.1) 
plt.ylabel('Actual Label') 
plt.xlabel('Predicted Label')
plt.show()

# Confusion Matrix Evaluation Metrics
print("Accuracy:", metrics.accuracy_score(y_test, y_pred)) 
print("Precision:", metrics.precision_score(y_test, y_pred)) 
print("Recall:", metrics.recall_score(y_test, y_pred))


# Confusion Matrix:
#  [[119  11]
#  [ 26  36]]
# Accuracy: 0.8072916666666666
# Precision: 0.7659574468085106
# Recall: 0.5806451612903226