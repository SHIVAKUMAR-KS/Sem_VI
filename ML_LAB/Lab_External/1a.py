import numpy as np
from scipy import stats
data = [10, 15, 21, 20, 18, 30, 25, 18, 18]
mean = np.mean(data)
median = np.median(data)
mode = stats.mode(data, keepdims=True).mode[0]
variance = np.var(data)
std_dev = np.std(data)
print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)
print("Variance:", variance)
print("Standard Deviation:", std_dev)