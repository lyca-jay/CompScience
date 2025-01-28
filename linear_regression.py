import numpy as np
import matplotlib.pyplot as plt

# Sample data
x_val = np.array([2, 4, 5, 7, 8])
y_val = np.array([3, 5, 7, 6, 8])

# Calculate the means of x and y
x_mean = np.mean(x_val)
y_mean = np.mean(y_val)

# Compute slope (m) and y-intercept (b) for linear equation y = mx + b
numerator = np.sum((x_val - x_mean) * (y_val - y_mean))
denominator = np.sum((x_val - x_mean) ** 2)
slope = numerator / denominator
y_intercept = y_mean - (slope * x_mean)

print(f"Slope (m): {slope}")
print(f"Intercept (b): {y_intercept}")

# Predicted values
y_pred = slope * x_val + y_intercept

# Plot data and regression line
plt.scatter(x_val, y_val, color='green', label='Observed Data')
plt.plot(x_val, y_pred, color='red', label='Regression Line')
plt.title('Manual Operation Linear Regression')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.legend()
plt.show()