def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	# Your code here
	for row in matrix:
		for j in range(len(row)):
			row[j] = row[j] * scalar
	return matrix