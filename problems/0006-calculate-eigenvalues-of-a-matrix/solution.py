import math
def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	a,b = matrix[0]
	c,d = matrix[1]
	trace = a+d
	det = a*d-b*c
	num1 = (trace+math.sqrt((trace**2)-4*det))/2
	num2 = (trace-math.sqrt((trace**2)-4*det))/2
	eigenvalues = []
	eigenvalues.append(num1)
	eigenvalues.append(num2)
	return eigenvalues