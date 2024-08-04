import numpy as np
import matplotlib.pyplot as plt

# Locally Weighted Regression (LWR) Function
def locally_weighted_regression(x_query, X, y, tau=0.1):
    X = np.array(X)
    y = np.array(y)
    x_query = np.array(x_query)
    
    # Gaussian kernel function
    kernel_weights = np.exp(-(X - x_query)**2 / (2 * tau**2))
    
    W = np.diag(kernel_weights)
    
    # Add an intercept term to X for the constant coefficient
    X_design = np.vstack([X, np.ones_like(X)]).T
    
    # Perform locally weighted linear regression
    theta = np.linalg.inv(X_design.T @ W @ X_design) @ (X_design.T @ W @ y)
    
    # Prediction at x_query
    y_query = np.array([x_query, 1]).T @ theta
    
    return y_query

# Generate Synthetic Data
np.random.seed(0)
X = np.linspace(0, 10, 100)
y = np.sin(X) + np.random.normal(0, 0.1, X.shape)

# Apply LWR and Predict
x_queries = np.linspace(0, 10, 100)
y_pred = [locally_weighted_regression(x, X, y, tau=0.5) for x in x_queries]

# Plot Results
plt.figure(figsize=(6, 4))
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(x_queries, y_pred, color='red', label='LWR Fit (tau=0.5)')
plt.title('Locally Weighted Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
