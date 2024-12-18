# Linear Regression from Scratch

This repository contains a simple implementation of a Linear Regression model built from scratch using Python. The project demonstrates how linear regression works through gradient descent optimization, and provides utilities to train the model, make predictions, evaluate performance, and save the model parameters.

## Features

- **Fit**: Trains the model using gradient descent with specified learning rate and epochs.
- **Predict**: Makes predictions based on input features.
- **Evaluation**: Computes the Root Mean Squared Error (RMSE) between predicted and actual values.
- **Model Saving**: Save the trained model as a binary file or save just the model parameters (weights and bias) to a JSON file.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/mohamedfarag22/linear-regression-scratch.git
    ```

2. Navigate to the project directory:
    ```bash
    cd linear-regression-scratch
    ```

3. Install the required dependencies (if any):
    ```bash
    pip install numpy
    ```

## Usage

You can train the model and make predictions using the following methods:

### 1. Train the model

```python
import numpy as np
from linear_regression import LinearRegression

# Sample data (features and target)
X = np.array([[1, 2], [2, 3], [4, 5]])
y = np.array([5, 7, 11])

# Initialize and fit the model
model = LinearRegression()
model.fit(X, y, lr=0.01, epoch=500)

# Model weights and bias after training
print("Weights:", model.weights)
print("Bias:", model.bias)
