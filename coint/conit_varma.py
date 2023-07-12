import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.stattools import coint

# Generate example VARMA time series data
np.random.seed(0)

# Parameters for VARMA process
n = 200  # Number of time steps
m = 2  # Number of variables (time series)

# Coefficients for VARMA process
phi = np.array([[0.6, -0.4], [0.3, -0.2]])
theta = np.array([[0.2, -0.3], [-0.1, 0.4]])

# Generate VARMA process data
errors = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=n)
series = np.zeros((n, m))
for t in range(2, n):
    series[t] = np.dot(phi, series[t-1]) + np.dot(theta, series[t-2]) + errors[t]

# Split the series into two time series
series1 = series[:, 0]
series2 = series[:, 1]

# Perform cointegration analysis
score, pvalue, _ = coint(series1, series2)

# Plot the two series
plt.plot(series1, label='Series 1')
plt.plot(series2, label='Series 2')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Cointegration Analysis')
plt.legend()

# Add cointegration test result to the plot
plt.text(0.05, 0.95, f'Cointegration p-value: {pvalue:.4f}', transform=plt.gca().transAxes)

# Show the plot
plt.show()