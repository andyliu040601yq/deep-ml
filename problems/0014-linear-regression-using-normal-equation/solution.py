import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
	# return theta
	X = np.array(X, dtype=float)
	y = np.array(y, dtype=float)

	X_temp = X.T @ X
	X_temp = np.linalg.inv(X_temp)
	theta = X_temp @ X.T @ y
	return np.round(theta,4).tolist()