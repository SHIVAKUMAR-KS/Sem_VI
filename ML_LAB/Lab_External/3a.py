import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
import matplotlib.pyplot as plt
df = pd.read_csv("Real-estate.csv")
df = df.drop(columns=['No'])
x = df.drop(columns=['Y house price of unit area'])
y = df['Y house price of unit area']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
print(y_predict)
print("mean_squared_error:", mean_squared_error(y_test, y_predict))
print("mean_absolute_error:", mean_absolute_error(y_test, y_predict))
print("R-Square:", r2_score(y_test, y_predict))
m, b = np.polyfit(y_test, y_predict, 1)
plt.plot(y_test, m*y_test + b, color='red', label="Best Fit Line")
plt.scatter(y_test, y_predict, color='blue', label="Predictions")
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.show()
