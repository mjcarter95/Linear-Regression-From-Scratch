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
    df = np.random.multivariate_normal(mean, cov, n)
    df = pd.DataFrame({'x':df[:,0],'y':df[:,1]})
    return df

# Calculate the gradient (m) and intercept (c) and return the line of best fit
def olsLinearRegression(x, y):
    print(x)
    print(y)

df = generateDataSet(100)
print(df.x)
#olsLinearRegression(x, y)

# Generate scatter graph using pyplot
plt.scatter(df.x, df.y)
plt.show()
