# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### 1. 1: Load and Normalize DataX = (X - mean(X)) / std(X)

### 2: Initialize Parametersm = 0b = 0learning_rate = αepochs = Nn = number of samples

### 3: Predict Outputy_pred = m * X + b

### 4: Compute Gradientsdm = (-2/n) * Σ [ X * (y - y_pred) ]db = (-2/n) * Σ [ (y - y_pred) ]

### 5: Update Parameters (repeat for each epoch)m = m - α * dmb = b - α * db
 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: TEAJESH R
RegisterNumber: 212225240167
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("Startup.csv")

# Select one feature (R&D Spend) and target (Profit)
X = data['R&D Spend'].values
y = data['Profit'].values

# Normalize (important for gradient descent)
X = (X - X.mean()) / X.std()

# Initialize parameters
m = 0
b = 0

learning_rate = 0.01
epochs = 1000
n = len(X)

# Gradient Descent
for i in range(epochs):
    y_pred = m * X + b
    
    # Gradients
    dm = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)
    
    # Update
    m = m - learning_rate * dm
    b = b - learning_rate * db

print("Slope (m):", m)
print("Intercept (b):", b)

# Predictions for plotting
y_pred = m * X + b

# Plot
plt.scatter(X, y)
plt.plot(X, y_pred)

plt.xlabel("R&D Spend (Normalized)")
plt.ylabel("Profit")
plt.title("Gradient Descent on 50_Startups Dataset")

plt.show()
```

## Output:
<img width="458" height="339" alt="TEAJESH ML 3" src="https://github.com/user-attachments/assets/26cb351c-c02a-4934-a5ee-1276d5252781" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
