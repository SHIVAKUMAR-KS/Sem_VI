import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize, stats, interpolate
from scipy.fft import fft
 
# 1. Integration of sin(x) from 0 to pi
def f(x):
	return np.sin(x)
integration_result, integration_error = integrate.quad(f, 0, np.pi)
print("Integration result:", integration_result)
 
# 2. Minimize a quadratic function
def quadratic(x):
	return x**2 - 4*x + 3
opt_result = optimize.minimize(quadratic, 0)
print("Minimum of quadratic function at x =", opt_result.x[0])
 
# 3. FFT of a sine wave signal Fast Fourier Transform (FFT)
t = np.linspace(0, 1, 100)
signal = np.sin(2 * np.pi * 5 * t)
fft_result = fft(signal)
print("FFT result (magnitude):", np.abs(fft_result[:10]))
 
# 4. Statistical analysis on normal distribution
data = np.random.normal(0, 1, 1000)
mean = np.mean(data)
std_dev = np.std(data)
print("Mean:", mean)
print("Standard Deviation:", std_dev)
cdf_value = stats.norm.cdf(0, loc=0, scale=1)
print("CDF at x = 0:", cdf_value) #Cumulative Distribution Function
 
# 5. Interpolation example
x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 1, 4, 9, 16])
f_interp = interpolate.interp1d(x, y, kind='quadratic')
y_interp = f_interp(2.5)
print("Interpolated value at x = 2.5:", y_interp)
 
x_new = np.linspace(0, 4, 100)
y_new = f_interp(x_new)
 
plt.plot(x, y, 'o', label="Original data")
plt.plot(x_new, y_new, '-', label="Interpolated curve")
plt.legend()
plt.show()