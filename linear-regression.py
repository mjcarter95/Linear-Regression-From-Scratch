# OLS Linear Regression from scratch in python
# Approximate the line y = mx + c for a given data set of y and x

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate a random data set of n size using np.random.multivariate_normal
def generateDataSet(n):
    mean = [1,1]
    cov = [[0.5,0.3], [0.3,0.3]]
    x = np.random.multivariate_normal(mean, cov, n)
    y = np.random.multivariate_normal(mean, cov, n)
    return x, y

# Calculate the gradient (m) and intercept (c) and return the line of best fit
def olsLinearRegression(x, y):
    print(x)
    print(y)

x, y = generateDataSet(100)
olsLinearRegression(x, y)

# Generate scatter graph using pyplot
plt.scatter(x, y)
plt.show()
