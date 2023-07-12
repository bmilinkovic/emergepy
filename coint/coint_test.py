import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint

# Generate two example time series data
np.random.seed(0)
n = 100
time = np.arange(n)
series1 = np.cumsum(np.random.randn(n))
series2 = 2 * series1 + np.random.randn(n)

# Check cointegration between the two series
score, pvalue, _ = coint(series1, series2)

# Plot the two series
plt.plot(time, series1, label='Series 1')
plt.plot(time, series2, label='Series 2')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Cointegration between Series 1 and Series 2')
plt.legend()

# Add cointegration test result to the plot
plt.text(0.05, 0.95, f'Cointegration p-value: {pvalue:.4f}', transform=plt.gca().transAxes)

# Show the plot
plt.show()