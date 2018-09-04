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
def olsLinearRegression(data):
    m = ((df["x"].mean() * df["y"].mean()) - (df["x"] * df["y"]).mean()) / ((df["x"].mean())**2 - (df["x"]**2).mean())
    c = df["y"].mean() - m * df["x"].mean()
    return m, c

df = generateDataSet(250)
m, c = olsLinearRegression(df)

print("Line of best fit: y = "+str(m)+"x +", c)

# Generate scatter graph using pyplot
plt.scatter(df.x, df.y)
plt.show()
